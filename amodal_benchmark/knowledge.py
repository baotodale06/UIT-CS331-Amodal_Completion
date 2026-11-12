"""Initialize project notes without overwriting an existing research record."""
from datetime import date
from pathlib import Path
import re


def init_notes(directory, project):
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", project):
        raise ValueError("Project must be a lowercase, hyphen-separated identifier")
    templates = {
        "plan.md": "## Objective\n\n## Dataset and protocol\n\n## Baseline\n\n## Next hypothesis\n\n## Validation\n",
        "progress.md": "## Completed\n\n## Running experiments\n\n## Next steps\n",
        "discoveries.md": "## Validated findings\n\nRecord evidence, run paths, protocol and limitations.\n",
        "closed_ideas.md": "## Scoped negative results\n\nRecord the tested hypothesis, evidence and conditions for revisiting it.\n",
        "session.md": "## Current task\n\n## Constraints\n\n## Active work\n\n## Checks and evidence\n\n## Handoff\n",
    }
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    created = []
    for name, body in templates.items():
        try:
            with (root / name).open("x", encoding="utf-8") as stream:
                stream.write(f"# {project}: {name[:-3]}\n\nInitialized {date.today().isoformat()}.\n\n" + body)
            created.append(name)
        except FileExistsError:
            pass
    return created
