"""Install the current review UI in an existing frozen image packet."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

UI_FILES = ("index.html", "style.css", "core.js", "app.js")


def install(packet, site):
    packet, site = Path(packet), Path(site)
    public = packet / "dist"
    public.mkdir(parents=True, exist_ok=True)
    # The website's frozen study files are recovered before the large image ZIP.
    for name in ("study.json", "study.js"):
        if not (public / name).exists():
            shutil.copyfile(site / "dist" / name, public / name)
    study = json.loads((public / "study.json").read_bytes())
    key = json.loads((packet / "COORDINATOR_ONLY/key.json").read_bytes())
    receipt = json.loads((packet / "COORDINATOR_ONLY/build.json").read_bytes())
    if (key["study_id"] != study["study_id"]
            or hashlib.sha256((public / "study.json").read_bytes()).hexdigest() != receipt["study_sha256"]):
        raise ValueError("Study identity or manifest hash changed")
    mappings = {row["case_id"]: row for row in key["cases"]}
    hashes = {case["files"][name]: sha for case in study["cases"]
              for name, sha in mappings[case["id"]]["display_sha256"].items()}
    script = "window.PREFERENCE_HASHES = " + json.dumps(hashes) + ";\n"
    for directory in (public, site / "dist"):
        (directory / "integrity.js").write_text(script, encoding="utf-8")
    for name in UI_FILES:
        shutil.copyfile(site / "dist" / name, public / name)
    launcher = '@echo off\r\nstart "" "%~dp0dist\\index.html"\r\n'
    (packet / "Start review.cmd").write_bytes(launcher.encode("utf-8"))
    print(json.dumps({"packet": str(packet), "study_id": study["study_id"], "image_hashes": len(hashes)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-dir", required=True, type=Path)
    p.add_argument("--site-dir", type=Path, default=Path(__file__).resolve().parents[1] / "sites/amodal-preference")
    args = p.parse_args()
    install(args.packet_dir, args.site_dir)
