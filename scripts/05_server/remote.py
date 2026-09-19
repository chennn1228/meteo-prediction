"""AutoDL SSH/SFTP helper using a local password file (password never printed)."""
import argparse
import os
import stat
import sys
from pathlib import Path

import paramiko

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
HOST = "connect.westb.seetacloud.com"
PORT = 21410
USER = "root"
PWD_FILE = ROOT / "secrets" / "autodl_password.txt"


def client():
    pwd = PWD_FILE.read_text(encoding="utf-8").strip()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=pwd, timeout=30,
              banner_timeout=30, auth_timeout=30)
    return c


def run(cmd):
    c = client()
    stdin, stdout, stderr = c.exec_command(cmd, timeout=3600, get_pty=False)
    for line in stdout:
        print(line, end="")
    err = stderr.read().decode(errors="replace")
    if err:
        print(err, end="", file=sys.stderr)
    code = stdout.channel.recv_exit_status()
    c.close()
    return code


def mkdirs(sftp, path):
    cur = ""
    for p in [x for x in path.split("/") if x]:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def put(local, remote):
    c = client()
    sftp = c.open_sftp()
    mkdirs(sftp, str(Path(remote).parent))
    sftp.put(str(local), str(remote))
    c.close()
    print("uploaded", local, "->", remote)


def get(remote, local):
    c = client()
    sftp = c.open_sftp()
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    size = sftp.stat(str(remote)).st_size
    off = Path(local).stat().st_size if Path(local).exists() else 0
    if off > size:
        off = 0
    mode = "ab" if off else "wb"
    with sftp.open(str(remote), "rb") as rf, open(str(local), mode) as lf:
        rf.seek(off)
        while off < size:
            data = rf.read(4 * 1024 * 1024)
            if not data:
                break
            lf.write(data)
            off += len(data)
    c.close()
    print("downloaded", remote, "->", local, "bytes", off)


def puttree(local_dir, remote_dir, max_mb=None):
    c = client()
    sftp = c.open_sftp()
    mkdirs(sftp, remote_dir)
    total = 0
    base = remote_dir.rstrip("/")
    for p in sorted(Path(local_dir).rglob("*")):
        rel = p.relative_to(local_dir).as_posix()
        if p.is_dir():
            mkdirs(sftp, f"{base}/{rel}")
        if p.is_file():
            if max_mb and p.stat().st_size > max_mb * 1024 * 1024:
                print("skip large", p)
                continue
            dst = f"{base}/{rel}"
            mkdirs(sftp, dst.rsplit("/", 1)[0])
            sftp.put(str(p), dst)
            total += p.stat().st_size
            print("uploaded", rel)
    c.close()
    print("total bytes", total)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("command")
    p = sub.add_parser("put")
    p.add_argument("local")
    p.add_argument("remote")
    g = sub.add_parser("get")
    g.add_argument("remote")
    g.add_argument("local")
    t = sub.add_parser("puttree")
    t.add_argument("local")
    t.add_argument("remote")
    t.add_argument("--max-mb", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "check":
        code = run("whoami; hostname; df -h /root/autodl-tmp; "
                   "mkdir -p /root/autodl-tmp/meteo_prediction; "
                   "echo ok > /root/autodl-tmp/meteo_prediction/write_test; "
                   "cat /root/autodl-tmp/meteo_prediction/write_test; "
                   "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    elif a.cmd == "run":
        code = run(a.command)
    elif a.cmd == "put":
        put(a.local, a.remote); code = 0
    elif a.cmd == "get":
        get(a.remote, a.local); code = 0
    else:
        puttree(a.local, a.remote, a.max_mb); code = 0
    raise SystemExit(code)
