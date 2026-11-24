"""Repair a Sites Git HTTP 413 using bounded pushes of an already frozen tree.

Linux-only transport helper. Credentials enter through hidden stdin, stay in
memory, and are supplied to Git in an environment variable. No asset is rebuilt.
The official Sites source helper still verifies and packages the final checkout.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import termios

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "sites/amodal-preference"
STATE = ROOT / "output/human-preference-20261122/site-tools/upload-progress.json"
MAX_BYTES = 24 * 1024 * 1024

settings = termios.tcgetattr(sys.stdin.fileno())
hidden = list(settings)
hidden[3] &= ~termios.ECHO
termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, hidden)
print("Ready for Sites credential JSON on stdin (input is hidden).", flush=True)
try:
    credential = json.loads(sys.stdin.readline())
finally:
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, settings)
if credential["auth_mode"] != "http_extra_header":
    raise ValueError("Expected the existing Sites bearer-header transport")
token, remote, branch = credential["token"], credential["remote_url"], credential["branch"]
if not remote.startswith("https://") or "\n" in token:
    raise ValueError("Invalid source credential")
env = os.environ.copy()
env["SITES_GIT_AUTHORIZATION"] = "Authorization: Bearer " + token
env["GIT_INDEX_FILE"] = str(STATE.with_suffix(".index"))
env["GIT_TERMINAL_PROMPT"] = "0"
base = ["git", "-c", "user.name=Sites", "-c", "user.email=sites@users.noreply.openai.com"]
auth = ["-c", "credential.helper=", "-c", "http.extraHeader=", "-c", "http.followRedirects=false",
        f"--config-env=http.{remote}.extraHeader=SITES_GIT_AUTHORIZATION"]


def git(*args, data=None, network=False):
    command = base + (auth if network else []) + list(args)
    result = subprocess.run(command, cwd=SITE, env=env, input=data, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        detail = result.stderr.decode(errors="replace").replace(token, "[redacted]")
        raise RuntimeError(detail)
    return result.stdout


def remote_head():
    result = git("ls-remote", "--heads", remote, f"refs/heads/{branch}", network=True).decode().split()
    return result[0] if result else None


def save(state):
    STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def push(sha):
    git("push", remote, f"{sha}:refs/heads/{branch}", network=True)
    if remote_head() != sha:
        raise ValueError("Remote head did not match the uploaded batch")


manifest = json.loads((SITE / ".openai/hosting.json").read_bytes())
if manifest["project_id"] != "appgprj_6b02fe6d29848191b670dd42ed2ab89d":
    raise ValueError("Wrong Site checkout")
full_head = git("rev-parse", "HEAD").decode().strip()
full_tree = git("rev-parse", "HEAD^{tree}").decode().strip()
head = remote_head()
if STATE.exists():
    state = json.loads(STATE.read_bytes())
    if state["full_tree"] != full_tree:
        raise ValueError("Frozen source tree changed; reconcile before resuming")
    pending = state.get("pending")
    if pending:
        if head == state["head"]:
            push(pending["head"])
        elif head != pending["head"]:
            raise ValueError("Remote source changed independently")
        state["head"], state["tree"] = pending["head"], pending["tree"]
        state["completed"].extend(pending["names"])
        state.pop("pending")
        save(state)
    elif head != state["head"]:
        raise ValueError("Remote source changed independently")
    git("read-tree", state["tree"])
else:
    state = {"full_head": full_head, "full_tree": full_tree, "head": head,
             "completed": [], "project_id": manifest["project_id"]}
    git("read-tree", "--empty")
    state["tree"] = git("write-tree").decode().strip()
    save(state)
entries = []
for item in git("ls-tree", "-r", "-z", state["full_head"]).split(b"\0"):
    if not item:
        continue
    fields, raw_name = item.split(b"\t", 1)
    mode, kind, oid = fields.decode().split()
    name = raw_name.decode("utf-8")
    if kind != "blob":
        raise ValueError("Unexpected non-file source entry")
    entries.append((name, mode, oid, (SITE / name).stat().st_size))
entries.sort(key=lambda e: (e[0].startswith("dist/assets/"), e[0]))
completed = set(state["completed"])
remaining = [e for e in entries if e[0] not in completed]
batches, current, size = [], [], 0
for entry in remaining:
    if current and size + entry[3] > MAX_BYTES:
        batches.append(current); current, size = [], 0
    current.append(entry); size += entry[3]
if current:
    batches.append(current)
for number, batch in enumerate(batches, 1):
    index_data = b"".join(f"{mode} {oid}\t{name}\0".encode() for name, mode, oid, _ in batch)
    git("update-index", "-z", "--index-info", data=index_data)
    tree = git("write-tree").decode().strip()
    args = ["commit-tree", tree]
    if state["head"]:
        args += ["-p", state["head"]]
    sha = git(*args, data=b"Upload frozen preference study assets\n").decode().strip()
    state["pending"] = {"head": sha, "tree": tree, "names": [e[0] for e in batch]}
    save(state)
    push(sha)
    state["head"], state["tree"] = sha, tree
    state["completed"].extend(state.pop("pending")["names"])
    save(state)
    if number == 1 or number % 5 == 0 or number == len(batches):
        print(json.dumps({"batch": number, "batches": len(batches), "files": len(state["completed"]),
                          "total_files": len(entries)}), flush=True)
if state["tree"] != state["full_tree"] or remote_head() != state["head"]:
    raise ValueError("Final tree or remote head mismatch")
# Retain the original all-at-once commit for audit; update only this new Site's branch.
git("update-ref", "refs/heads/before-batched-upload", state["full_head"])
git("update-ref", f"refs/heads/{branch}", state["head"], full_head)
print(json.dumps({"status": "uploaded", "project_id": manifest["project_id"],
                  "commit_sha": state["head"], "source_tree": state["tree"],
                  "files": len(entries)}), flush=True)
