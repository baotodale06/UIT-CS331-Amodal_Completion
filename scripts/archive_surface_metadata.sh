#!/usr/bin/env bash
# Package JSON evidence only; keep scientific PNGs in their original run.
set -Eeuo pipefail
ARCHIVE="${1:?Archive destination required}"
shift
[[ "$#" -gt 0 && ! -e "$ARCHIVE" ]] || exit 2
for DIRECTORY in "$@"; do [[ -d "$DIRECTORY" ]] || exit 2; done
find "$@" -type f -name '*.json' -print0 | tar --null -czf "$ARCHIVE" -T -
sha256sum "$ARCHIVE"
