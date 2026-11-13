"""Content-addressed jobs, private checkpoints and non-destructive exports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__


class TaskBusyError(RuntimeError):
    """Another live process owns this task."""


class RetryRequiredError(RuntimeError):
    """A failed job requires the caller's explicit retry flag."""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_dict(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def fingerprint(source: Path, config: dict, source_sha256: str | None = None) -> str:
    content = {
        "source_sha256": source_sha256 or digest(source),
        "config": config,
        "version": __version__,
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


@contextmanager
def task_lock(path: Path):
    """Publish a complete PID lock atomically; reap only demonstrably dead owners."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="ascii") as handle:
            os.chmod(temporary, 0o600)
            handle.write(str(os.getpid()))
        for _ in range(3):
            try:
                # An exclusive hard link exposes the fully written PID in one operation.
                os.link(temporary, path)
                owned_inode = path.stat().st_ino
                break
            except FileExistsError:
                try:
                    stat = path.stat()
                    owner = int(path.read_text().strip())
                    if owner < 1:
                        raise ValueError("Invalid lock")
                    os.kill(owner, 0)
                except ProcessLookupError:
                    try:
                        if path.stat().st_ino == stat.st_ino:
                            path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                except FileNotFoundError:
                    continue
                except (ValueError, PermissionError):
                    pass
                raise TaskBusyError("Task is locked; inspect its lock before retrying") from None
        else:
            raise TaskBusyError("Task lock could not be acquired")
    finally:
        temporary.unlink(missing_ok=True)
    try:
        yield
    finally:
        try:
            if path.stat().st_ino == owned_inode:
                path.unlink()
        except FileNotFoundError:
            pass


def _status(job_dir: Path, task_id: str, status: str, progress: float, error_code=None):
    record = {"task_id": task_id, "status": status, "progress": progress, "error_code": error_code}
    atomic_json(job_dir / "job.json", record)
    event = dict(record, at=datetime.now(timezone.utc).isoformat())
    logfile = job_dir / "events.jsonl"
    fd = os.open(logfile, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def record_progress(job_dir: Path, completed_chunks: int, total_chunks: int) -> None:
    """Update bounded chunk progress without adding content or per-chunk event logs."""
    path = job_dir / "job.json"
    if not path.is_file():
        return  # The core pipeline is also usable without the task manager.
    previous = json.loads(path.read_text(encoding="utf-8"))
    fraction = min(1.0, max(0.0, completed_chunks / max(1, total_chunks)))
    atomic_json(
        path,
        {
            "task_id": previous["task_id"],
            "status": "running",
            "progress": round(0.9 * fraction, 6),
            "error_code": None,
        },
    )


def _new_attempt(output_dir: Path, source: Path, task_id: str) -> Path:
    stem = re.sub(r"[^\w.-]+", "_", source.stem, flags=re.UNICODE)[:80] or "media"
    parent = output_dir / f"{stem}-{task_id[:12]}"
    parent.mkdir(parents=True, exist_ok=True)
    for number in range(1, 100000):
        target = parent / f"attempt-{number:03d}"
        try:
            target.mkdir(mode=0o700)
            return target
        except FileExistsError:
            continue
    raise FileExistsError("Export attempt limit reached")


def _valid_artifacts(manifest: dict, output_dir: Path) -> bool:
    if manifest.get("output_root") != str(output_dir):
        return False
    files = manifest.get("files", [])
    if not files:
        return False
    try:
        return all(
            Path(item["path"]).is_file() and digest(Path(item["path"])) == item["sha256"]
            for item in files
        )
    except (OSError, KeyError, TypeError):
        return False


def run_task(source: Path, config: dict, output_dir: Path, retry: bool = False) -> dict:
    from ..core.paths import state_dir
    from ..core.pipeline import transcribe
    from ..exporters import export_transcript, validate_transcript

    source = source.expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("Input must be a regular file")
    output_dir = output_dir.expanduser().resolve()
    source_checksum = digest(source)
    task_id = fingerprint(source, config, source_checksum)
    job_dir = state_dir() / "jobs" / task_id
    job_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with task_lock(job_dir / "task.lock"):
        status_file = job_dir / "job.json"
        previous = _read_json_dict(status_file)
        manifest_file = job_dir / "artifacts.json"
        manifest = _read_json_dict(manifest_file)
        cached = job_dir / "transcript.json"
        if previous.get("status") == "completed" and _valid_artifacts(manifest, output_dir):
            return {
                "task_id": task_id,
                "status": "completed",
                "duplicate": True,
                "input_file": str(source),
                "outputs": [entry["path"] for entry in manifest["files"]],
                "metadata": manifest.get("metadata", {}),
            }
        if previous.get("status") == "failed" and not retry:
            error = RetryRequiredError("Use --retry to resume a failed task")
            error.task_id = task_id
            raise error
        _status(job_dir, task_id, "running", 0)
        try:
            if cached.is_file() and manifest.get("transcript_sha256") == digest(cached):
                transcript = json.loads(cached.read_text(encoding="utf-8"))
                validate_transcript(transcript)
            else:
                transcript = transcribe(source, config, job_dir)
                if digest(source) != source_checksum:
                    atomic_json(job_dir / "input-integrity-invalid.json", {"invalid": True})
                    raise ValueError("Source file changed during transcription")
                validate_transcript(transcript)
                atomic_json(cached, transcript)
            _status(job_dir, task_id, "exporting", 0.95)
            target = _new_attempt(output_dir, source, task_id)
            files = export_transcript(transcript, target, config["formats"])
            manifest = {
                "output_root": str(output_dir),
                "transcript_sha256": digest(cached),
                "files": [{"path": str(p.resolve()), "sha256": digest(p)} for p in files],
                "metadata": transcript["metadata"],
            }
            atomic_json(manifest_file, manifest)
            _status(job_dir, task_id, "completed", 1)
            return {
                "task_id": task_id,
                "status": "completed",
                "duplicate": False,
                "input_file": str(source),
                "outputs": [entry["path"] for entry in manifest["files"]],
                "metadata": transcript["metadata"],
                "warnings": transcript.get("warnings", []),
            }
        except KeyboardInterrupt:
            _status(job_dir, task_id, "interrupted", 0, "KeyboardInterrupt")
            raise
        except Exception as error:
            error.task_id = task_id
            _status(job_dir, task_id, "failed", 0, type(error).__name__)
            raise
