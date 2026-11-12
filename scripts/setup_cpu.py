#!/usr/bin/env python3
"""Create the locked Python 3.12 CPU environment without touching the GPU cache."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv-cpu",
                        help="Dedicated CPU environment (default: repository/.venv-cpu)")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Run this script with Python 3.12; see .python-version for the tested patch.")
    # venv records the base executable's directory as its stdlib home. Resolve
    # version-manager symlinks first (e.g. ~/.local/bin/python3.12 from uv).
    base_executable = Path(sys._base_executable)
    if base_executable != base_executable.resolve():
        subprocess.run([str(base_executable.resolve()), str(Path(__file__).resolve()),
                        *sys.argv[1:]], check=True)
        return
    destination = args.venv.resolve()
    marker = destination / ".amodal-cpu-environment"
    if destination.exists() and any(destination.iterdir()) and not marker.is_file():
        parser.error(f"Refusing to modify an environment not created by this script: {destination}")
    if not marker.exists():
        venv.EnvBuilder(with_pip=True, symlinks=sys.platform != "win32").create(destination)
        marker.write_text("amodal-benchmark CPU development only\n", encoding="utf-8")
    executable = destination / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([str(executable), "-c",
                    "import sys; assert sys.version_info[:2] == (3, 12), 'Expected Python 3.12'"],
                   check=True)
    for lock in ("build.lock", "cpu.lock"):
        subprocess.run([str(executable), "-m", "pip", "install", "--require-hashes",
                        "--only-binary=:all:", "-r", str(ROOT / "requirements" / lock)], check=True)
    subprocess.run([str(executable), "-m", "pip", "install", "--no-deps",
                    "--no-build-isolation", "-e", str(ROOT)], check=True)
    subprocess.run([str(executable), "-m", "pip", "check"], check=True)
    print(f"CPU environment ready: {destination}")
    print("Activate it, then run: bash scripts/check.sh (Linux/WSL)")


if __name__ == "__main__":
    main()
