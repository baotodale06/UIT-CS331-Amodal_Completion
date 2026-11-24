"""Prepare the compact hosted UI; full native images stay in the offline packet."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "sites/amodal-preference"
FILES = ("index.html", "style.css", "app.js", "core.js", "study.json", "study.js", "integrity.js")


def prepare():
    destination = SITE / "out"
    destination.mkdir(exist_ok=True)
    for name in FILES:
        shutil.copyfile(SITE / "dist" / name, destination / name)
    page = (destination / "index.html").read_text(encoding="utf-8")
    page = page.replace('<script defer src="study.js"></script>',
                        '<script>window.PREFERENCE_LOCAL_IMAGES = true;</script>\n  <script defer src="study.js"></script>')
    (destination / "index.html").write_text(page, encoding="utf-8")
    manifest_path = SITE / ".openai/hosting.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["static"]["directory"] = "out"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if {path.name for path in destination.iterdir()} != set(FILES):
        raise ValueError("Unexpected hosted reviewer files")
    print(json.dumps({"hosted": str(destination), "files": len(FILES), "images": "local_folder"}))


if __name__ == "__main__":
    prepare()
