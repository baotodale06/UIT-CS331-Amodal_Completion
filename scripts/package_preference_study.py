"""Package a frozen reviewer packet, with no coordinator key in its ZIP."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import zipfile


def package(folder, archive):
    folder, archive = Path(folder), Path(archive)
    if archive.exists():
        raise ValueError("Archive already exists")
    sizes = defaultdict(int)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for path in sorted((folder / "dist").rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(folder / "dist").as_posix())
                sizes[path.stem.split("-")[-1]] += path.stat().st_size
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "groups_bytes": dict(sizes)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-dir", required=True)
    p.add_argument("--archive", required=True)
    a = p.parse_args()
    package(a.packet_dir, a.archive)
