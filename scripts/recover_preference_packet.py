"""Resume the verified offline image ZIP and install/validate the local reviewer."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.install_preference_ui import install
from scripts.validate_preference_study import validate

PACKET = ROOT / "output/human-preference-20261122"
ARCHIVE = PACKET / "reviewer-images.zip"
PARTIAL = PACKET / "reviewer-images.zip.partial"
EXPECTED_BYTES = 1465779913
EXPECTED_SHA = "8199604eb08ea4cec5011b853efa34a78dbd1af614cf30015297808a36cc7ea0"


def status(phase, **values):
    record = {"phase": phase, **values}
    (PACKET / "local-recovery.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record), flush=True)


def recover():
    if ARCHIVE.exists() and ARCHIVE.stat().st_size != EXPECTED_BYTES:
        if PARTIAL.exists():
            raise ValueError("Both partial and incomplete final ZIPs exist; reconcile manually")
        ARCHIVE.replace(PARTIAL)
    if not ARCHIVE.exists():
        batch = PACKET / "offline-transfer.sftp"
        batch.write_text('reget /home/daoqm/Vu_Hung/cv_nangcao/output/human-preference-20261122/reviewer-images.zip "' +
                         PARTIAL.as_posix() + '"\n', encoding="utf-8")
        status("downloading", bytes_present=PARTIAL.stat().st_size if PARTIAL.exists() else 0, expected_bytes=EXPECTED_BYTES)
        subprocess.run(["sftp", "-oBatchMode=yes", "-oConnectTimeout=15", "-b", str(batch),
                        "daoqm@slurm.csse.rose-hulman.edu"], check=True)
        if PARTIAL.stat().st_size != EXPECTED_BYTES:
            raise ValueError("Offline ZIP length differs")
        PARTIAL.replace(ARCHIVE)
    status("verifying_archive")
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != EXPECTED_SHA:
        raise ValueError("Offline ZIP checksum differs")
    study = json.loads((PACKET / "dist/study.json").read_bytes())
    allowed = {relative for case in study["cases"] for relative in case["files"].values()}
    status("extracting", images=len(allowed))
    with zipfile.ZipFile(ARCHIVE) as archive:
        image_names = {info.filename for info in archive.infolist() if info.filename.startswith("assets/")}
        if image_names != allowed:
            raise ValueError("Offline ZIP image inventory differs")
        for name in sorted(allowed):
            target = (PACKET / "dist" / name).resolve()
            if (PACKET / "dist").resolve() not in target.parents:
                raise ValueError("Unsafe image path")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
    # Keep the current tested UI; the image archive's earlier UI is not installed.
    install(PACKET, ROOT / "sites/amodal-preference")
    status("validating_images")
    receipt = validate(PACKET)
    (PACKET / "local-validation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    status("ready", launcher=str(PACKET / "Start review.cmd"), **receipt)


if __name__ == "__main__":
    try:
        recover()
    except Exception as error:
        status("failed", error=str(error))
        raise
