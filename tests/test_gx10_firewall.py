"""Static checks for scripts/gx10-firewall/ (boot-time nftables firewall).

No root, no nft, no services: only reads the files and runs `bash -n`.
The live firewall is never touched. Mutation harness: tests/mutate_gx10_firewall.py.
"""
import ipaddress
import re
import subprocess
from pathlib import Path

import pytest

FW = Path(__file__).resolve().parent.parent / "scripts" / "gx10-firewall"
EXPECTED_PORTS = {5678, 6333, 6334, 8000, 8001}
NFT = FW / "gx10_fw.nft"
SERVICE = FW / "gx10-firewall.service"
ALERT_SERVICE = FW / "gx10-firewall-alert.service"
INSTALL = FW / "fw-install.sh"
SH_SCRIPTS = ["fw-install.sh", "fw-arm-revert.sh", "fw-blocktest.sh",
              "gx10-fw-alert", "gx10-fw-record"]
ALL_FILES = ["gx10_fw.nft", "gx10-firewall.service", "gx10-firewall-alert.service",
             "README.txt"] + SH_SCRIPTS


def read(name):
    return (FW / name).read_text(encoding="utf-8")


def code_lines(text):
    """Lines without full-line comments (so comments cannot satisfy or trip a check)."""
    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def set_elements(text, set_name):
    m = re.search(r"set\s+%s\s*\{.*?elements\s*=\s*\{([^}]*)\}" % set_name, text, re.S)
    assert m, "set %s not found" % set_name
    return {e.strip() for e in m.group(1).split(",") if e.strip()}


def port_sets(text):
    found = []
    for m in re.finditer(r"(?:dport|proto-dst)\s*\{([^}]*)\}", "\n".join(code_lines(text))):
        found.append({int(p) for p in re.findall(r"\d+", m.group(1))})
    return found


def chain_block(text, name):
    m = re.search(r"chain\s+%s\s*\{(.*?)\n  \}" % name, text, re.S)
    assert m, "chain %s not found" % name
    return m.group(1)


# Addresses that may appear anywhere in the firewall files. Rule-based on purpose:
# anything else (including documentation ranges such as 203.0.113.0/24 or
# 2001:db8::/32, which Python calls "private") counts as a public address.
ALLOWED_V4 = [ipaddress.ip_network(n) for n in (
    "127.0.0.0/8", "192.168.1.0/24", "172.17.0.0/16",
    "172.30.0.0/24",  # fw-blocktest.sh: throw-away Docker net that must be OUTSIDE the allow list
)]
ALLOWED_V6 = [ipaddress.ip_network(n) for n in ("::1/128", "fe80::/10")]
V4_RE = re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2})?(?![\d.])")
V6_RE = re.compile(r"(?<![0-9A-Za-z:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?:/\d{1,3})?")


def disallowed_addresses(text):
    bad = []
    for tok in V4_RE.findall(text):
        try:
            net = ipaddress.ip_network(tok, strict=False)
        except ValueError:
            bad.append(tok)
            continue
        if not any(net.subnet_of(a) for a in ALLOWED_V4):
            bad.append(tok)
    for tok in V6_RE.findall(text):
        try:
            net = ipaddress.ip_network(tok, strict=False)
        except ValueError:
            if "::" in tok:  # looks like an IPv6 address but unparseable: reject
                bad.append(tok)
            continue  # e.g. clock times such as 12:41:45
        if not any(net.subnet_of(a) for a in ALLOWED_V6):
            bad.append(tok)
    return bad


# ---------- rules file ----------

def test_table_name_and_no_flush():
    code = "\n".join(code_lines(read("gx10_fw.nft")))
    assert re.search(r"^table inet gx10_fw\s*\{", code, re.M)
    assert not re.search(r"flush\s+ruleset", code)


def test_replace_is_add_delete_recreate_in_one_file():
    lines = code_lines(read("gx10_fw.nft"))
    assert lines[0] == "add table inet gx10_fw"
    assert lines[1] == "delete table inet gx10_fw"


@pytest.mark.parametrize("chain", ["input", "forward"])
def test_chain_priority_is_minus_10_and_policy_accept(chain):
    block = chain_block(read("gx10_fw.nft"), chain)
    assert re.search(r"hook %s priority -10;\s*policy accept;" % chain, block)


def test_port_set_is_exactly_the_five_ports_everywhere():
    sets = port_sets(read("gx10_fw.nft"))
    # input: v4 accept, v6 accept, drop; forward: ct original proto-dst -> 4 occurrences
    assert len(sets) == 4
    for s in sets:
        assert s == EXPECTED_PORTS


def test_lan4_only_private_v4_ranges():
    assert set_elements(read("gx10_fw.nft"), "lan4") == {
        "127.0.0.0/8", "192.168.1.0/24", "172.17.0.0/16"}


def test_lan6_only_loopback_and_link_local():
    assert set_elements(read("gx10_fw.nft"), "lan6") == {"::1", "fe80::/10"}


def test_forward_drops_dnat_from_outside_lan():
    fwd = chain_block(read("gx10_fw.nft"), "forward")
    m = [ln for ln in fwd.splitlines() if "ct status dnat" in ln]
    assert len(m) == 1
    assert "ip saddr != @lan4" in m[0]
    assert m[0].rstrip().endswith("drop")
    assert "ct original proto-dst" in m[0]


def test_ssh_accepted_only_from_lan_sets_then_dropped():
    inp = chain_block(read("gx10_fw.nft"), "input")
    ssh = [ln.strip() for ln in inp.splitlines() if "dport 22 " in ln]
    assert ssh == [
        "tcp dport 22 ip saddr @lan4 accept",
        "tcp dport 22 ip6 saddr @lan6 accept",
        "tcp dport 22 counter drop",
    ]


def test_input_established_and_loopback_accepted_before_drops():
    inp = chain_block(read("gx10_fw.nft"), "input")
    lines = [ln.strip() for ln in inp.splitlines()]
    est = lines.index("ct state established,related accept")
    first_drop = min(i for i, ln in enumerate(lines) if ln.endswith("drop"))
    assert est < first_drop
    assert 'iifname "lo" accept' in lines


def test_port_rules_accept_lan_before_drop():
    inp = chain_block(read("gx10_fw.nft"), "input")
    rows = [ln.strip() for ln in inp.splitlines() if "dport {" in ln]
    assert len(rows) == 3
    assert rows[0].endswith("ip saddr @lan4 accept")
    assert rows[1].endswith("ip6 saddr @lan6 accept")
    assert rows[2].endswith("counter drop")


def test_no_public_addresses_in_rules():
    code = "\n".join(code_lines(read("gx10_fw.nft")))
    assert disallowed_addresses(code) == []
    assert "0.0.0.0" not in code and "::/0" not in code


def test_address_checker_rejects_public_and_documentation_ranges():
    for bad in ("203.0.113.7", "8.8.8.8", "0.0.0.0/0", "10.0.0.1", "192.168.2.1",
                "2001:db8::/32", "2001:4860::1", "::/0", "2001:db8::"):
        assert disallowed_addresses("x %s y" % bad) == [bad], bad
    for good in ("127.0.0.0/8", "192.168.1.128", "172.17.0.0/16", "::1", "fe80::/10",
                 "12:41:45", "port 8000"):
        assert disallowed_addresses("x %s y" % good) == [], good


# ---------- service files ----------

def unit_lines(name):
    return [ln.strip() for ln in code_lines(read(name))]


def test_service_ordering_and_failure_handling():
    u = unit_lines("gx10-firewall.service")
    assert any(ln.startswith("Before=") and "network-pre.target" in ln.split("=", 1)[1].split()
               for ln in u)
    assert "OnFailure=gx10-firewall-alert.service" in u
    assert "DefaultDependencies=no" in u
    assert not any(ln.startswith("Requires=") for ln in u)  # must not drag Docker/SSH down


def test_service_type_and_remain_after_exit():
    u = unit_lines("gx10-firewall.service")
    assert "Type=oneshot" in u
    assert "RemainAfterExit=yes" in u


def test_service_precheck_and_load_use_same_rules_file():
    u = unit_lines("gx10-firewall.service")
    pre = [ln for ln in u if ln.startswith("ExecStartPre=")]
    assert pre == ["ExecStartPre=/usr/sbin/nft -c -f /etc/gx10-fw/gx10_fw.nft"]
    assert "ExecStart=/usr/sbin/nft -f /etc/gx10-fw/gx10_fw.nft" in u
    assert "ExecReload=/usr/sbin/nft -f /etc/gx10-fw/gx10_fw.nft" in u


def test_service_record_step_is_optional_with_dash():
    """Deliberate: without the leading '-', a failing record script fails the
    service and triggers ExecStop, which deletes the table."""
    u = unit_lines("gx10-firewall.service")
    post = [ln for ln in u if ln.startswith("ExecStartPost=")]
    assert post == ["ExecStartPost=-/usr/local/sbin/gx10-fw-record loaded"]


def test_service_stop_deletes_only_our_table_and_is_optional():
    """ExecStop is deliberate (stop removes only inet gx10_fw)."""
    u = unit_lines("gx10-firewall.service")
    stop = [ln for ln in u if ln.startswith("ExecStop=")]
    assert stop == ["ExecStop=-/usr/sbin/nft delete table inet gx10_fw"]


def test_service_never_flushes_ruleset():
    for name in ("gx10-firewall.service", "gx10-firewall-alert.service"):
        assert not re.search(r"flush\s+ruleset", "\n".join(code_lines(read(name))))


def test_alert_service_runs_alert_script():
    u = unit_lines("gx10-firewall-alert.service")
    assert "ExecStart=/usr/local/sbin/gx10-fw-alert" in u
    assert "Type=oneshot" in u


# ---------- scripts ----------

@pytest.mark.parametrize("name", SH_SCRIPTS)
def test_script_syntax_ok(name):
    r = subprocess.run(["bash", "-n", str(FW / name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("name", SH_SCRIPTS)
def test_script_has_bash_shebang(name):
    assert read(name).startswith("#!/bin/bash")


def test_install_has_no_flush_ruleset_command():
    for ln in code_lines(read("fw-install.sh")):
        assert not re.match(r"\s*(?:\S+\s*&&\s*)?nft\s+(-\S+\s+)*flush\b", ln), ln


def test_install_refuses_rules_containing_flush_ruleset():
    code = "\n".join(code_lines(read("fw-install.sh")))
    assert re.search(r"grep -qE 'flush\[\[:space:\]\]\+ruleset' .*&& die", code)


def test_install_checks_nftables_service_state():
    code = "\n".join(code_lines(read("fw-install.sh")))
    assert "systemctl is-enabled nftables.service" in code
    assert re.search(r"enabled\|enabled-runtime\|linked\|alias\) die", code)


def test_install_never_enables_or_starts_services():
    for ln in code_lines(read("fw-install.sh")):
        assert not re.match(r"\s*(?:\S+\s*&&\s*)?systemctl\s+(enable|start|restart|reload|mask|unmask)\b", ln), ln
        assert not re.search(r"(?<!is-)\bsystemctl\s+(enable|start)\b", ln.split("die", 1)[0]), ln


def test_install_has_uninstall_that_requires_stopped_disabled_service():
    code = "\n".join(code_lines(read("fw-install.sh")))
    assert '"${1:-}" = "--uninstall"' in code
    assert re.search(r"enabled\|enabled-runtime\) die", code)
    assert re.search(r'"\$ac" = "active" \] && die', code)


def test_install_installs_the_five_artifacts_with_root_ownership():
    code = "\n".join(code_lines(read("fw-install.sh")))
    for f in ("gx10-firewall.service", "gx10-firewall-alert.service",
              "gx10-fw-record", "gx10-fw-alert", "gx10_fw.nft"):
        assert re.search(r"install \$OWN -m \d+ \"\$HERE/%s\"" % re.escape(f), code), f
    assert 'OWN=""; [ -z "$D" ] && OWN="-o root -g root"' in code


def test_record_and_alert_scripts_always_exit_zero():
    for name in ("gx10-fw-record", "gx10-fw-alert"):
        assert code_lines(read(name))[-1].strip() == "exit 0"


def test_arm_revert_deletes_only_our_table_and_needs_root():
    code = "\n".join(code_lines(read("fw-arm-revert.sh")))
    assert "nft delete table inet gx10_fw" in code
    assert '"$(id -u)" = 0' in code
    assert "flush" not in code


def test_blocktest_covers_all_five_ports():
    assert 'PORTS="22 8000 6333 8001 5678"' in read("fw-blocktest.sh")


# ---------- privacy ----------

@pytest.mark.parametrize("name", ALL_FILES)
def test_no_secrets_or_public_ips(name):
    text = read(name)
    assert not re.search(r"hf_[A-Za-z0-9]{10,}", text)
    assert not re.search(r"(sk|pk)-lf-[0-9a-f-]{8,}", text)
    assert not re.search(r"(?i)(password|passwd|token)\s*[=:]\s*\S{4,}", text.replace("不含任何密碼或 token", ""))
    assert "HF_TOKEN" not in text
    assert disallowed_addresses(text) == [], name


def test_no_private_state_files_copied_in():
    names = {p.name for p in FW.iterdir()}
    assert names == set(ALL_FILES)
