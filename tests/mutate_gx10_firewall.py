#!/usr/bin/env python3
"""Mutation harness for tests/test_gx10_firewall.py.

Works on a temporary copy of the repo (scripts/gx10-firewall + the test); the
real files and the live firewall are never touched. Each mutation must make
the test file fail. Usage: python3 tests/mutate_gx10_firewall.py
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def sub(old, new, count=1):
    def f(t):
        assert old in t, "mutation anchor missing: %r" % old
        return t.replace(old, new, count)
    return f

M = [
 ("拿掉 ExecStartPost 的 '-'", "gx10-firewall.service", sub("ExecStartPost=-", "ExecStartPost=")),
 ("nft 規則加入 flush ruleset", "gx10_fw.nft", sub("add table inet gx10_fw\n", "flush ruleset\nadd table inet gx10_fw\n")),
 ("拿掉 5678（四處全拿掉）", "gx10_fw.nft", sub(", 5678 }", " }", 4)),
 ("只拿掉 forward 鏈的 6334", "gx10_fw.nft", lambda t: t.replace("{ 8000, 8001, 6333, 6334, 5678 } ip saddr != @lan4", "{ 8000, 8001, 6333, 5678 } ip saddr != @lan4")),
 ("5678 換成 5679（四條規則）", "gx10_fw.nft", lambda t: "\n".join(l if l.lstrip().startswith("#") else l.replace("5678", "5679") for l in t.split("\n"))),
 ("只有 input 的 v6 accept 把 5678 換成 5679", "gx10_fw.nft", lambda t: t.replace("{ 8000, 8001, 6333, 6334, 5678 } ip6 saddr @lan6 accept", "{ 8000, 8001, 6333, 6334, 5679 } ip6 saddr @lan6 accept")),
 ("input priority -10 改 0", "gx10_fw.nft", sub("hook input priority -10", "hook input priority 0")),
 ("forward priority -10 改 0", "gx10_fw.nft", sub("hook forward priority -10", "hook forward priority 0")),
 ("lan4 加入 0.0.0.0/0", "gx10_fw.nft", sub("172.17.0.0/16 }", "172.17.0.0/16, 0.0.0.0/0 }")),
 ("lan4 加入公網 IP 203.0.113.7（文件範例位址）", "gx10_fw.nft", sub("172.17.0.0/16 }", "172.17.0.0/16, 203.0.113.7 }")),
 ("lan6 加入公網 IPv6（2001:db8::/32）", "gx10_fw.nft", sub("fe80::/10 }", "fe80::/10, 2001:db8::/32 }")),
 ("拿掉 forward 的 drop", "gx10_fw.nft", sub("ip saddr != @lan4 counter drop", "ip saddr != @lan4 counter accept")),
 ("forward 拿掉 ct status dnat", "gx10_fw.nft", sub("ct status dnat ", "")),
 ("forward 的 != 改成 ==", "gx10_fw.nft", sub("ip saddr != @lan4", "ip saddr == @lan4")),
 ("SSH 22 不再丟棄", "gx10_fw.nft", sub("tcp dport 22 counter drop\n", "")),
 ("SSH 22 對所有來源放行", "gx10_fw.nft", sub("tcp dport 22 ip saddr @lan4 accept", "tcp dport 22 accept")),
 ("移除 Before=", "gx10-firewall.service", lambda t: re.sub(r"^Before=.*\n", "", t, flags=re.M)),
 ("移除 OnFailure=", "gx10-firewall.service", lambda t: re.sub(r"^OnFailure=.*\n", "", t, flags=re.M)),
 ("移除 ExecStartPre 的 nft -c", "gx10-firewall.service", lambda t: re.sub(r"^ExecStartPre=.*\n", "", t, flags=re.M)),
 ("移除 ExecStop", "gx10-firewall.service", lambda t: re.sub(r"^ExecStop=.*\n", "", t, flags=re.M)),
 ("移除 RemainAfterExit", "gx10-firewall.service", sub("RemainAfterExit=yes\n", "")),
 ("service 加 Requires=docker.service", "gx10-firewall.service", sub("[Service]", "Requires=docker.service\n\n[Service]")),
 ("fw-install.sh 拿掉 nftables.service 檢查", "fw-install.sh", sub("enabled|enabled-runtime|linked|alias) die", "never) die")),
 ("fw-install.sh 加入 systemctl enable", "fw-install.sh", sub("  systemctl daemon-reload || echo", "  systemctl enable gx10-firewall.service\n  systemctl daemon-reload || echo")),
 ("fw-install.sh 加入 systemctl start", "fw-install.sh", sub("  systemctl daemon-reload || echo", "  systemctl start gx10-firewall.service\n  systemctl daemon-reload || echo")),
 ("fw-install.sh 拿掉 --uninstall", "fw-install.sh", sub('"${1:-}" = "--uninstall"', '"${1:-}" = "--nope"')),
 ("fw-install.sh 拿掉 flush 檢查", "fw-install.sh", sub("&& die \"規則檔含有 flush ruleset", "&& true \"規則檔含有 flush ruleset")),
 ("fw-install.sh 加入 nft flush ruleset", "fw-install.sh", sub('echo "== 5/5 安裝"', 'nft flush ruleset\necho "== 5/5 安裝"')),
 ("fw-install.sh 語法錯誤", "fw-install.sh", lambda t: t + "\nif then fi (\n"),
 ("gx10-fw-record 語法錯誤", "gx10-fw-record", lambda t: t + "\nfi fi\n"),
 ("gx10-fw-record 最後不再 exit 0", "gx10-fw-record", lambda t: t.rstrip()[:-len("exit 0")] + "exit 1\n"),
 ("gx10-fw-alert 最後不再 exit 0", "gx10-fw-alert", lambda t: t.rstrip()[:-len("exit 0")] + "exit 1\n"),
 ("fw-arm-revert.sh 改成 flush", "fw-arm-revert.sh", sub("nft delete table inet gx10_fw", "nft flush ruleset")),
 ("README 混入公網 IPv4", "README.txt", lambda t: t + "\n連線來源 203.0.113.7\n"),
 ("blocktest 混入公網 IPv6", "fw-blocktest.sh", lambda t: t + "\n# 2001:db8::1\n"),
 ("service 檔混入公網 IPv4", "gx10-firewall.service", lambda t: t + "\n# 203.0.113.7\n"),
 ("install 腳本混入不在清單的私有網段", "fw-install.sh", lambda t: t + "\n# 10.0.0.5\n"),
 ("blocktest 混入 token 值", "fw-blocktest.sh", lambda t: t + "\nexport HF_TOKEN=hf_abcdefghijklmnop\n"),
 ("blocktest 少測 5678", "fw-blocktest.sh", sub('PORTS="22 8000 6333 8001 5678"', 'PORTS="22 8000 6333 8001"')),
 ("混入不該複製的檔案", "commands-to-paste.md", lambda t: "x"),
]

def main():
    ok = 0
    survivors = []
    for label, fname, fn in M:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "tests").mkdir()
            shutil.copytree(ROOT / "scripts" / "gx10-firewall", td / "scripts" / "gx10-firewall")
            shutil.copy(ROOT / "tests" / "test_gx10_firewall.py", td / "tests")
            target = td / "scripts" / "gx10-firewall" / fname
            if fname == "commands-to-paste.md":
                target.write_text("x", encoding="utf-8")
            else:
                target.write_text(fn(target.read_text(encoding="utf-8")), encoding="utf-8")
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                str(td / "tests" / "test_gx10_firewall.py")], capture_output=True, text=True)
            killed = r.returncode != 0
            ok += killed
            if not killed:
                survivors.append(label)
            print("%s  %s" % ("抓到" if killed else "沒抓到", label))
    print("\n抓到 %d／總共 %d" % (ok, len(M)))
    if survivors:
        print("沒抓到：", survivors)
    sys.exit(0 if not survivors else 1)

if __name__ == "__main__":
    main()
