"""Prepare an isolated three-case UI fixture from a downloaded ZIP prefix."""
import hashlib
import json
from pathlib import Path
import shutil
import struct

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "output/human-preference-20261122"
site = ROOT / "sites/amodal-preference/dist"
qa = base / "verification/browser/dist"
(qa / "assets").mkdir(parents=True, exist_ok=True)
study = json.loads((site / "study.json").read_bytes())
study["study_id"] = "qa-" + study["study_id"]
study["cases"] = study["cases"][:3]
study["eligible"] = 3
keys = json.loads((base / "COORDINATOR_ONLY/key.json").read_bytes())["cases"][:3]
hashes = {case["files"][name]: sha for case, key in zip(study["cases"], keys)
          for name, sha in key["display_sha256"].items()}
needed = set(hashes)
# ZIP_STORED entries have known lengths in the local header. No extraction of
# untrusted paths; only the allowlisted first three cases and verified bytes.
with (base / "reviewer-images.zip").open("rb") as stream:
    while needed:
        header = stream.read(30)
        sig, ver, flags, compression, mtime, mdate, crc, compressed, size, nlen, xlen = struct.unpack("<4s5H3I2H", header)
        assert sig == b"PK\x03\x04" and compression == 0 and not flags & 8
        name = stream.read(nlen).decode("utf-8")
        stream.read(xlen)
        data = stream.read(compressed)
        assert len(data) == compressed
        if name in needed:
            assert hashlib.sha256(data).hexdigest() == hashes[name]
            (qa / name).write_bytes(data)
            needed.remove(name)
for name in ("index.html", "app.js", "core.js", "style.css"):
    shutil.copyfile(site / name, qa / name)
(qa / "study.json").write_text(json.dumps(study), encoding="utf-8")
(qa / "study.js").write_text("window.PREFERENCE_STUDY = " + json.dumps(study) + ";", encoding="utf-8")
(qa / "integrity.js").write_text("window.PREFERENCE_HASHES = " + json.dumps(hashes) + ";", encoding="utf-8")
print("Isolated three-case QA packet ready; no production ratings.")
