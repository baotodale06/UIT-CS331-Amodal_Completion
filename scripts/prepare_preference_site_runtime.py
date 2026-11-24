"""Provision an isolated Node helper runtime and stage the existing image packet."""
import hashlib
from pathlib import Path
import shutil
import tarfile
import urllib.request

root = Path(__file__).resolve().parents[1]
destination = root / "output/human-preference-20261122/site-tools"
destination.mkdir(parents=True, exist_ok=True)
version = "v22.16.0"
filename = f"node-{version}-linux-x64.tar.xz"
url = f"https://nodejs.org/dist/{version}/"
archive = destination / filename
urllib.request.urlretrieve(url + filename, archive)
sums = urllib.request.urlopen(url + "SHASUMS256.txt", timeout=60).read().decode()
expected = next(line.split()[0] for line in sums.splitlines() if line.split()[-1] == filename)
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise ValueError("Node runtime hash mismatch")
with tarfile.open(archive) as source:
    for member in source.getmembers():
        target = (destination / member.name).resolve()
        if destination.resolve() not in target.parents:
            raise ValueError("Unexpected runtime archive path")
    source.extractall(destination)
site = root / "sites/amodal-preference"
(site / ".openai").mkdir(parents=True, exist_ok=True)
if not (site / "dist").exists():
    shutil.copytree(root / "output/human-preference-20261122/dist", site / "dist")
print({"runtime": str(destination / f"node-{version}-linux-x64/bin/node"), "site": str(site)})
