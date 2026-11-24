"""Verify the byte counts and SHA-256 hashes of published research evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "results/published-files.json").read_bytes())
    if manifest.get("schema") != 1 or not manifest.get("files"):
        raise SystemExit("Invalid or empty published-results manifest")
    seen = set()
    total = 0
    for row in manifest["files"]:
        relative = row["path"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root / "output") or relative in seen:
            raise SystemExit(f"Unsafe or duplicate evidence path: {relative}")
        seen.add(relative)
        if not path.is_file():
            raise SystemExit(f"Missing published evidence: {relative}")
        data = path.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise SystemExit(f"Published evidence does not match its manifest: {relative}")
        total += len(data)
    print(f"Verified {len(seen)} published result files ({total:,} bytes).")
    print("Scope: file integrity only; no new inference or benchmark success claim.")


if __name__ == "__main__":
    main()
