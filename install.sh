#!/usr/bin/env bash
# Isolated source install. No sudo and no system-Python package writes.
set -euo pipefail
umask 077
repo_root="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
download_models=1
run_smoke=1
for arg in "$@"; do
    case "$arg" in
        --skip-models) download_models=0 ;;
        --skip-smoke) run_smoke=0 ;;
        -h|--help)
            printf '%s\n' 'Usage: bash install.sh [--skip-models] [--skip-smoke]' \
              'Default: install isolated runtime, download default models, run CPU smoke, link Skill.'
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
    esac
done
printf 'System: %s %s\n' "$(uname -s)" "$(uname -m)"
case "$(uname -s)" in
    Darwin) cache_root="${LMT_CACHE_DIR:-$HOME/Library/Caches/local-media-transcriber}" ;;
    *) cache_root="${LMT_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/local-media-transcriber}" ;;
esac
mkdir -p "$cache_root/uv"
export UV_CACHE_DIR="$cache_root/uv"
if ! command -v uv >/dev/null 2>&1; then
    # Bootstrap only in an app-owned cache directory, never in system Python.
    if ! command -v python3 >/dev/null 2>&1; then
        printf '%s\n' 'Install Python 3 and uv from their official installers, then rerun.' >&2
        exit 1
    fi
    bootstrap_dir="$cache_root/installer-venv"
    if [[ -e "$bootstrap_dir" && ! -f "$bootstrap_dir/pyvenv.cfg" ]]; then
        printf '%s\n' 'Installer cache exists but is not a virtual environment; refusing to overwrite.' >&2
        exit 1
    fi
    python3 -m venv "$bootstrap_dir"
    "$bootstrap_dir/bin/python" -m pip install 'uv==0.10.7'
    uv_cmd="$bootstrap_dir/bin/uv"
else
    uv_cmd="$(command -v uv)"
fi
"$uv_cmd" --version
"$uv_cmd" python install 3.11
for command_name in ffmpeg ffprobe; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Missing %s.\n' "$command_name" >&2
        if [[ "$(uname -s)" == Darwin ]] && command -v brew >/dev/null 2>&1; then
            printf '%s\n' 'Installing FFmpeg through the existing Homebrew installation.'
            brew install ffmpeg
        else
            printf '%s\n' 'Install FFmpeg with your OS package manager, ensure ffprobe is on PATH, then rerun.' >&2
            exit 1
        fi
        break
    fi
done
"$uv_cmd" sync --frozen --python 3.11 --extra asr --extra docx --extra dev
runtime="$repo_root/.venv/bin"
"$runtime/python" -c 'from local_media_transcriber.core.compat import install_audio_only_adapter; install_audio_only_adapter(); import funasr, torch; print("FunASR import OK; torch", torch.__version__, "CPU threads", torch.get_num_threads())'
if (( download_models )); then
    "$runtime/transcribe" setup --download
else
    "$runtime/transcribe" setup
fi
"$runtime/transcribe" doctor
if (( run_smoke )); then
    smoke_dir="$(mktemp -d "$cache_root/install-smoke.XXXXXXXX")"
    trap 'rm -rf -- "$smoke_dir"' EXIT
    "$runtime/python" "$repo_root/examples/smoke.py" --work-dir "$smoke_dir"
fi
skill_source="$repo_root/.agents/skills/local-media-transcription"
skill_target="$HOME/.agents/skills/local-media-transcription"
mkdir -p "$(dirname "$skill_target")"
if [[ -L "$skill_target" ]]; then
    resolved_target="$(cd -P -- "$skill_target" 2>/dev/null && pwd || true)"
    if [[ "$resolved_target" != "$skill_source" ]]; then
        printf 'Existing Skill link retained: %s\n' "$skill_target" >&2
        printf 'New Skill is ready at: %s\n' "$skill_source" >&2
        exit 1
    fi
elif [[ -e "$skill_target" ]]; then
    printf 'Existing Skill retained; refusing overwrite: %s\n' "$skill_target" >&2
    printf 'New Skill is ready at: %s\n' "$skill_source" >&2
    exit 1
else
    ln -s "$skill_source" "$skill_target"
fi
"$skill_target/scripts/transcribe-local.sh" --version
"$runtime/python" "$skill_target/scripts/verify_skill.py"
printf 'Installed CLI: %s/transcribe\nSkill: %s\n' "$runtime" "$skill_target"
printf '%s\n' 'Use: source .venv/bin/activate' \
    'Skill discovery status is reported above. No chat or model call was created.'
