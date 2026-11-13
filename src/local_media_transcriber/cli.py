"""CLI entry point. Model downloads are restricted to explicit setup/download commands."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__

SUPPORTED = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
}
MODES = ("transcription", "diarization", "subtitle", "learning", "research")
FORMATS = {"txt", "srt", "vtt", "json", "csv", "md", "docx"}
LANGUAGES = ("zh", "en", "auto", "yue", "ja", "ko")


def _emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _read_mapping(value, label):
    if value is None:
        return {}
    result = (
        value
        if isinstance(value, dict)
        else json.loads(Path(value).expanduser().read_text(encoding="utf-8"))
    )
    if not isinstance(result, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in result.items()
    ):
        raise ValueError(f"{label} must be a JSON object mapping strings to strings")
    return result


def _configuration(args, override=None):
    from .models.registry import list_models, select_model

    allowed = {
        "mode",
        "speakers",
        "language",
        "model",
        "no_diarization",
        "formats",
        "channel",
        "glossary",
        "speaker_map",
        "reading",
        "chunk_seconds",
        "threads",
        "time_offset_ms",
        "exercises",
    }
    values = {key: getattr(args, key) for key in allowed}
    if override:
        if not isinstance(override, dict) or set(override) - allowed:
            raise ValueError("Manifest contains unsupported configuration keys")
        values.update(override)
    if values["mode"] not in MODES or values["language"] not in LANGUAGES:
        raise ValueError("Unsupported mode or language")
    if str(values["channel"]) not in {"mix", "split", "0", "1"}:
        raise ValueError("channel must be mix, split, 0 or 1")
    for key in ("threads", "chunk_seconds"):
        if isinstance(values[key], bool) or not isinstance(values[key], int) or values[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if values["chunk_seconds"] > 600:
        raise ValueError("chunk_seconds must be <= 600 to bound memory usage")
    if not isinstance(values["time_offset_ms"], int) or isinstance(values["time_offset_ms"], bool):
        raise ValueError("time_offset_ms must be an integer")
    for key in ("reading", "no_diarization", "exercises"):
        if not isinstance(values[key], bool):
            raise ValueError(f"{key} must be boolean")
    if values["speakers"] is not None:
        count = values["speakers"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1 or count > 50:
            raise ValueError("speakers must be an integer from 1 to 50")
        if values["no_diarization"]:
            raise ValueError("--speakers conflicts with --no-diarization")
    formats = values.pop("formats")
    if isinstance(formats, str):
        formats = [item.strip() for item in formats.split(",") if item.strip()]
    if not isinstance(formats, list) or not formats or any(f not in FORMATS for f in formats):
        raise ValueError("Unsupported or empty formats list")
    if values["mode"] == "subtitle":
        formats = list(dict.fromkeys([*formats, "srt", "vtt", "txt"]))
    values["formats"] = sorted(set(formats))
    values["glossary"] = _read_mapping(values["glossary"], "glossary")
    values["speaker_map"] = _read_mapping(values["speaker_map"], "speaker_map")
    values["channel"] = str(values["channel"])
    values["diarization"] = not values["no_diarization"] and (
        values["speakers"] is not None or values["mode"] in {"diarization", "research"}
    )
    values["export_exercises"] = values.pop("exercises")
    selected = select_model(values["language"], values["model"])
    values["model"] = selected["name"]
    values["selected_model"] = selected
    values["model_versions"] = {item["name"]: item["version"] for item in list_models()}
    return values


def _common(parser):
    parser.add_argument("--mode", choices=MODES, default="transcription")
    parser.add_argument("--speakers", type=int)
    parser.add_argument("--language", choices=LANGUAGES, default="zh")
    parser.add_argument("--model")
    parser.add_argument("--no-diarization", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--formats", default="txt,json")
    parser.add_argument("--channel", choices=("mix", "split", "0", "1"), default="split")
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--speaker-map", type=Path)
    parser.add_argument("--reading", action="store_true")
    parser.add_argument("--chunk-seconds", type=int, default=60)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--time-offset-ms", type=int, default=0)
    parser.add_argument("--exercises", action="store_true")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Inference is always offline; download models explicitly first",
    )


def parser():
    root = argparse.ArgumentParser(prog="transcribe", description="Local audio/video transcription")
    root.add_argument(
        "--version", action="version", version=f"Local Media Transcriber {__version__}"
    )
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect dependencies without downloading anything")
    setup = commands.add_parser("setup", help="Initialize private cache/state directories")
    setup.add_argument("--download", action="store_true", help="Download default Chinese models")
    models = commands.add_parser("models", help="List or explicitly download registered models")
    model_commands = models.add_subparsers(dest="model_command")
    model_commands.add_parser("list")
    download = model_commands.add_parser("download")
    download.add_argument("name")
    cache = commands.add_parser("cache")
    cache.add_subparsers(dest="cache_command", required=True).add_parser("info")
    run = commands.add_parser("run", help="Transcribe one local media file")
    run.add_argument("input", type=Path)
    _common(run)
    batch = commands.add_parser("batch", help="Process a local folder; preserve successful jobs")
    batch.add_argument("input", type=Path)
    batch.add_argument("--recursive", action="store_true")
    batch.add_argument(
        "--manifest",
        type=Path,
        help="JSON object mapping relative media paths to configuration overrides",
    )
    _common(batch)
    return root


def _doctor():
    from .models.registry import cache_info

    command = [
        sys.executable,
        "-c",
        "from local_media_transcriber.core.runtime import offline_runtime, quiet_runtime\n"
        "with offline_runtime(), quiet_runtime():\n from local_media_transcriber.core.compat import install_audio_only_adapter\n install_audio_only_adapter()\n import funasr\n import torch\n",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    checks = {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "funasr_import": result.returncode == 0,
        "cache": cache_info(),
        "device": "cpu",
        "network_used": False,
    }
    _emit(checks)
    return 0 if checks["ffmpeg"] and checks["ffprobe"] and checks["funasr_import"] else 1


def _run(args):
    from .task_manager.manager import run_task

    source = args.input.expanduser().resolve(strict=True)
    if source.suffix.lower() not in SUPPORTED or not source.is_file():
        raise ValueError("Input must be a supported local media file")
    result = run_task(source, _configuration(args), args.output, retry=args.retry)
    _emit(result)
    return 0


def _batch(args):
    from .task_manager.manager import run_task

    source_dir = args.input.expanduser().resolve(strict=True)
    if not source_dir.is_dir():
        raise ValueError("Batch input must be a directory")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else {}
    if not isinstance(manifest, dict) or any(not isinstance(k, str) for k in manifest):
        raise ValueError("Manifest must be a JSON object")
    files = sorted(
        p
        for p in (source_dir.rglob("*") if args.recursive else source_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    relative = {p.relative_to(source_dir).as_posix() for p in files}
    if set(manifest) - relative:
        raise ValueError("Manifest includes a missing or unsupported media path")
    # Validate every override before starting any inference.
    prepared = [
        (p, _configuration(args, manifest.get(p.relative_to(source_dir).as_posix()))) for p in files
    ]
    results = []
    failed = 0
    for path, config in prepared:
        try:
            results.append(run_task(path, config, args.output, retry=args.retry))
        except Exception as error:
            failed += 1
            results.append(
                {
                    "status": "failed",
                    "error_code": type(error).__name__,
                    "input_file": str(path),
                    "task_id": getattr(error, "task_id", None),
                }
            )
    _emit(
        {
            "total": len(files),
            "succeeded": len(files) - failed,
            "failed": failed,
            "results": results,
        }
    )
    return 1 if failed or not files else 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor()
        if args.command == "setup":
            from .core.paths import cache_dir, state_dir

            for directory in (cache_dir(), state_dir()):
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            if args.download:
                from .models.registry import download_model

                for name in ("paraformer-zh", "fsmn-vad", "ct-punc"):
                    download_model(name)
            _emit(
                {
                    "cache": str(cache_dir()),
                    "state": str(state_dir()),
                    "downloaded_defaults": args.download,
                }
            )
            return 0
        if args.command == "models":
            from .models.registry import download_model, list_models

            if args.model_command == "download":
                _emit({"model": args.name, "path": str(download_model(args.name))})
            else:
                _emit(list_models())
            return 0
        if args.command == "cache":
            from .models.registry import cache_info

            _emit(cache_info())
            return 0
        return _run(args) if args.command == "run" else _batch(args)
    except KeyboardInterrupt:
        _emit({"status": "interrupted", "error_code": "KeyboardInterrupt"})
        return 130
    except Exception as error:
        # Upstream exception strings can contain filenames, transcripts or access tokens.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_code": type(error).__name__,
                    "task_id": getattr(error, "task_id", None),
                    "hint": "Run transcribe doctor and transcribe cache info; download missing models "
                    "explicitly with transcribe models download NAME; use --retry for failed tasks",
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
