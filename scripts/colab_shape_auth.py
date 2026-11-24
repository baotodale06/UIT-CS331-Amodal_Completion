"""Complete the official Colab CLI OAuth flow without putting secrets in logs."""
import argparse
from importlib import resources
import json
import os
from pathlib import Path


def main():
    from colab_cli.auth import PUBLIC_SCOPES, REMOTE_REDIRECT_URI
    from google_auth_oauthlib.flow import InstalledAppFlow

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("begin", "finish"))
    parser.add_argument("--code-file", type=Path)
    args = parser.parse_args()
    folder = Path.home() / ".config" / "shape-colab"
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    pending = folder / "pending.json"
    config = json.loads(resources.files("colab_cli").joinpath("oauth_config.json").read_text())
    if args.action == "begin":
        flow = InstalledAppFlow.from_client_config(config, PUBLIC_SCOPES)
        flow.redirect_uri = REMOTE_REDIRECT_URI
        url, state = flow.authorization_url(prompt="consent", token_usage="remote")
        with os.fdopen(os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as stream:
            json.dump(dict(state=state, code_verifier=flow.code_verifier), stream)
        print(url)
        return
    if args.code_file is None:
        parser.error("finish requires --code-file")
    state = json.loads(pending.read_text())
    flow = InstalledAppFlow.from_client_config(config, PUBLIC_SCOPES, **state)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    flow.fetch_token(code=args.code_file.read_text().strip())
    # Separate task credentials, never the user's existing ADC or repository files.
    credentials = folder / "credentials.json"
    payload = json.loads(flow.credentials.to_json())
    payload["type"] = "authorized_user"
    with os.fdopen(os.open(credentials, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as stream:
        json.dump(payload, stream)
    pending.unlink()
    print("Colab authorization saved outside the repository; no token was printed.")


if __name__ == "__main__":
    main()
