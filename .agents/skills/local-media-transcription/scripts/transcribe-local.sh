#!/usr/bin/env bash
set -euo pipefail
skill_root="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd -P -- "$skill_root/../../.." && pwd)"
if command -v transcribe >/dev/null 2>&1; then
    version="$(transcribe --version 2>/dev/null || true)"
    if [[ "$version" == *"Local Media Transcriber"* ]]; then
        exec transcribe "$@"
    fi
fi
if [[ -x "$repo_root/.venv/bin/transcribe" ]]; then
    exec "$repo_root/.venv/bin/transcribe" "$@"
fi
if [[ -x "$repo_root/.venv/Scripts/transcribe.exe" ]]; then
    exec "$repo_root/.venv/Scripts/transcribe.exe" "$@"
fi
printf '%s\n' "Local Media Transcriber runtime not found. Run bash install.sh in the repository checkout." >&2
exit 1
