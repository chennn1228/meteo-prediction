"""本地结果回收：服务器跑完 → 下载摘要包 → 解包到本地 → 通知服务器关机。

用法（本地项目根目录）：
    .venv\\Scripts\\python.exe scripts\\05_server\\local_collect.py          # 尝试回收一次
    .venv\\Scripts\\python.exe scripts\\05_server\\local_collect.py --check  # 只看服务器有没有出包
"""
import subprocess
import sys
import tarfile
import shutil
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
REMOTE = ROOT / "scripts" / "05_server" / "remote.py"
PY = sys.executable
REMOTE_TAR = "/root/autodl-tmp/meteo_results_summary.tar.gz"
STAGE = ROOT / "deploy" / "02_results"


def remote_run(cmd):
    r = subprocess.run([PY, str(REMOTE), "run", cmd], capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    check_only = "--check" in sys.argv
    code, out = remote_run(f"test -f {REMOTE_TAR} && echo READY || echo NOT_READY")
    ready = "READY" in out and "NOT_READY" not in out
    if not ready:
        print("[等待] 服务器结果包尚未生成")
        return 1
    print("[就绪] 服务器结果包已生成")
    if check_only:
        return 0

    STAGE.mkdir(parents=True, exist_ok=True)
    local_tar = STAGE / "meteo_results_summary.tar.gz"
    r = subprocess.run([PY, str(REMOTE), "get", REMOTE_TAR, str(local_tar)])
    if r.returncode != 0 or not local_tar.exists():
        print("[失败] 下载失败，请重试")
        return 1
    print(f"[完成] 已下载 {local_tar}  ({local_tar.stat().st_size/1024/1024:.1f} MB)")

    unpack = STAGE / "unpacked"
    if unpack.exists():
        shutil.rmtree(unpack)
    unpack.mkdir(parents=True)
    with tarfile.open(local_tar) as tf:
        tf.extractall(unpack)
    copied = []
    for sub in ("reports", "figs"):
        src = unpack / sub
        if src.exists():
            shutil.copytree(src, ROOT / sub, dirs_exist_ok=True)
            copied.append(sub)
    print(f"[完成] 已解包并覆盖到本地目录：{', '.join(copied)}（原始包保留在 deploy/02_results/）")

    code, out = remote_run("touch /root/autodl-tmp/DOWNLOAD_OK && echo OK")
    if "OK" in out:
        print("[完成] 已通知服务器：可以关机")
    else:
        print("[警告] 未确认到服务器信号，请手动检查")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
