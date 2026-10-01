"""Checks for scripts/gx10-backup-mac/ (the Mac-side backup pull script, parameterised for the public repo).

Static checks plus a few runs of the script against stub ssh/osascript programs: no network, no real ssh,
no notification. Mutation harness: tests/mutate_gx10_backup_mac.py.
"""
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

MAC = Path(__file__).resolve().parent.parent / "scripts" / "gx10-backup-mac"
FILES = ["gx10-backup.sh", "com.psf.gx10-backup.plist.example", "README.md"]
SH = MAC / "gx10-backup.sh"
PLIST = MAC / "com.psf.gx10-backup.plist.example"
README = MAC / "README.md"


def read(name):
    return (MAC / name).read_text(encoding="utf-8")


def line_starting(text, prefix):
    hits = [ln for ln in text.splitlines() if ln.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


# ---------- files and syntax ----------

def test_only_the_expected_files_exist():
    assert sorted(p.name for p in MAC.iterdir()) == sorted(FILES)


def test_script_syntax_ok():
    r = subprocess.run(["bash", "-n", str(SH)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ---------- nothing real in the public files ----------

@pytest.mark.parametrize("name", FILES)
def test_no_real_identifiers(name):
    text = read(name)
    assert "psf01" not in text
    assert "192.168." not in text
    assert not re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text), "no IPv4 address of any kind"
    assert not re.search(r"/Users/(?!YOUR_USER\b)", text)
    assert "/home/" not in text


# ---------- parameterisation ----------

def test_host_is_required_and_has_no_default():
    text = read("gx10-backup.sh")
    assert re.fullmatch(r'HOST="\$\{GX10_BACKUP_HOST:\?[^}]+\}"', line_starting(text, "HOST="))
    assert not re.search(r"GX10_BACKUP_HOST(:-|:=|=|-)", text)


def test_key_and_dest_defaults_and_overrides():
    text = read("gx10-backup.sh")
    assert line_starting(text, "KEY=") == 'KEY="${GX10_BACKUP_KEY:-$HOME/.ssh/gx10_backup}"'
    assert line_starting(text, "DEST=") == 'DEST="${GX10_BACKUP_DEST:-$HOME/GX10Backup}"'


# ---------- plist example ----------

def _plist_dict(elem):
    out, items = {}, list(elem)
    for k, v in zip(items[0::2], items[1::2]):
        assert k.tag == "key"
        if v.tag == "dict":
            out[k.text] = _plist_dict(v)
        elif v.tag == "array":
            out[k.text] = [x.text for x in v]
        elif v.tag in ("true", "false"):
            out[k.text] = v.tag == "true"
        elif v.tag == "integer":
            out[k.text] = int(v.text)
        else:
            out[k.text] = v.text
    return out


def test_plist_example_is_valid_xml_with_placeholders():
    root = ET.parse(PLIST).getroot()
    assert root.tag == "plist"
    d = _plist_dict(root.find("dict"))
    assert d["Label"] == "com.psf.gx10-backup"
    assert d["ProgramArguments"][0] == "/bin/bash"
    assert "YOUR_USER" in d["ProgramArguments"][1] and d["ProgramArguments"][1].endswith("/bin/gx10-backup.sh")
    assert d["EnvironmentVariables"] == {"GX10_BACKUP_HOST": "user@host"}
    assert d["StartCalendarInterval"] == {"Hour": 9, "Minute": 0}
    assert d["RunAtLoad"] is True


def test_readme_covers_setup_and_forced_command():
    text = read("README.md")
    for needle in ("GX10_BACKUP_HOST", "GX10_BACKUP_KEY", "GX10_BACKUP_DEST", "launchctl",
                   "gx10_backup_serve.py", "user@host", "YOUR_USER"):
        assert needle in text, needle


# ---------- behaviour against stubs (no network) ----------

def _stub(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def _run(tmp_path, env_extra, path_stub=True):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "ssh_calls.txt"
    _stub(bin_dir / "ssh", 'echo "$@" >> "%s"\nexit 255\n' % calls)
    osa = _stub(tmp_path / "osa", "exit 0\n")
    env = {"HOME": str(home), "PATH": "%s:/usr/bin:/bin" % bin_dir, "GX10_BACKUP_OSASCRIPT": str(osa)}
    env.update(env_extra)
    r = subprocess.run(["bash", str(SH)], env=env, capture_output=True, text=True, cwd=str(tmp_path), timeout=60)
    return r, home, calls


def test_missing_host_stops_before_touching_anything(tmp_path):
    r, home, calls = _run(tmp_path, {})
    assert r.returncode != 0
    assert "GX10_BACKUP_HOST" in r.stderr
    assert list(home.iterdir()) == [], "must not create the backup folder when HOST is not set"
    assert not calls.exists(), "must not attempt any ssh connection"


def test_host_and_dest_overrides_are_used(tmp_path):
    dest = tmp_path / "custom-dest"
    r, home, calls = _run(tmp_path, {"GX10_BACKUP_HOST": "tester@host.invalid", "GX10_BACKUP_DEST": str(dest)})
    assert r.returncode == 1, r.stderr
    assert (dest / "STATUS.txt").is_file()
    assert not (home / "GX10Backup").exists(), "default folder must not be used when DEST is set"
    text = calls.read_text()
    assert "tester@host.invalid latest" in text
    assert "tester@host.invalid failed" in text


@pytest.mark.parametrize("override", [False, True])
def test_default_and_overridden_key_are_passed_to_ssh(tmp_path, override):
    env = {"GX10_BACKUP_HOST": "tester@host.invalid"}
    key = str(tmp_path / "custom_key")
    if override:
        env["GX10_BACKUP_KEY"] = key
    r, home, calls = _run(tmp_path, env)
    expected = key if override else str(home / ".ssh" / "gx10_backup")
    assert "-i %s" % expected in calls.read_text()
    assert "-o BatchMode=yes" in calls.read_text()
