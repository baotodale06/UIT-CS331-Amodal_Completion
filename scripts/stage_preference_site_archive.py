"""Stage the exact pushed reviewer source using already verified local images.

Fetch only Git metadata and small UI files. Verify every Git blob identity before
the official Sites packager uses the replica; never copy coordinator material.
"""
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "output/human-preference-20261122"
STAGE = PACKET / "site-tools/local-source"
REMOTE = "/home/daoqm/Vu_Hung/cv_nangcao/sites/amodal-preference"
HOST = "daoqm@slurm.csse.rose-hulman.edu"


def ssh(command):
    return subprocess.run(
        ["ssh", "-oBatchMode=yes", "-oConnectTimeout=15", HOST,
         "cd " + shlex.quote(REMOTE) + " && " + command],
        check=True, stdout=subprocess.PIPE,
    ).stdout


def stage():
    receipt = json.loads((PACKET / "local-validation.json").read_bytes())
    if receipt["status"] != "passed":
        raise ValueError("Full local image validation is required")
    sha = ssh("git rev-parse HEAD").decode().strip()
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError("Invalid commit identity")
    entries = {}
    for record in ssh("git ls-tree -r -z " + sha).split(b"\0"):
        if not record:
            continue
        fields, name = record.split(b"\t", 1)
        mode, kind, blob = fields.decode().split()
        if mode != "100644" or kind != "blob":
            raise ValueError("Unexpected source entry type")
        entries[name.decode()] = blob
    small = [name for name in entries if not name.startswith("dist/assets/")]
    allowed = {".openai/hosting.json", "README.md"} | {
        "dist/" + name for name in
        ("index.html", "style.css", "app.js", "core.js", "study.json", "study.js", "integrity.js")
    }
    if "out/index.html" in entries:
        allowed |= {"out/" + name for name in
                    ("index.html", "style.css", "app.js", "core.js", "study.json", "study.js", "integrity.js")}
    if set(small) != allowed:
        raise ValueError("Unexpected reviewer source inventory")
    archive = ssh("git archive " + sha + " -- " + " ".join(map(shlex.quote, small)))
    STAGE.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or member.name not in allowed:
                raise ValueError("Unsafe source archive entry")
            target = STAGE / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.extractfile(member).read())
    study = json.loads((STAGE / "dist/study.json").read_bytes())
    images = {"dist/" + path for case in study["cases"] for path in case["files"].values()}
    if set(entries) != allowed | images:
        raise ValueError("Source assets differ from the frozen study")
    for name, expected in entries.items():
        target = (STAGE / name).resolve()
        if STAGE.resolve() not in target.parents:
            raise ValueError("Unsafe source path")
        if name in images:
            original = PACKET / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    target.hardlink_to(original)
                except OSError:
                    shutil.copyfile(original, target)
        data = target.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected:
            raise ValueError("Source blob differs: " + name)
    actual_files = {path.relative_to(STAGE).as_posix() for path in STAGE.rglob("*") if path.is_file()}
    if actual_files != set(entries):
        raise ValueError("Unexpected local staging files")
    result = {"status": "passed", "commit_sha": sha, "files": len(entries),
              "images": len(images), "source": str(STAGE)}
    (PACKET / "site-tools/local-source-validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    stage()
