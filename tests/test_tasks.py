"""Task-boundary tests use a fixture pipeline; they do not measure ASR accuracy."""

import json
import os
import sys
import types
from pathlib import Path

import pytest

from local_media_transcriber.task_manager.manager import (
    RetryRequiredError,
    TaskBusyError,
    digest,
    fingerprint,
    run_task,
    task_lock,
)


@pytest.fixture
def task_env(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setattr("local_media_transcriber.core.paths.state_dir", lambda: state)
    calls = []
    transcript = {
        "metadata": {
            "source_file": "fixture.wav",
            "duration_ms": 1000,
            "language": "zh",
            "model": "fixture-only",
        },
        "speakers": [],
        "segments": [
            {"id": 1, "start_ms": 0, "end_ms": 1000, "speaker_id": None, "text": "Fixture text"}
        ],
    }

    def fake_pipeline(source, config, job_dir):
        calls.append(job_dir)
        return transcript

    module = types.ModuleType("local_media_transcriber.core.pipeline")
    module.transcribe = fake_pipeline
    monkeypatch.setitem(sys.modules, "local_media_transcriber.core.pipeline", module)
    source = tmp_path / "private-source.wav"
    source.write_bytes(b"not-media-fixture-for-manager-only")
    return source, {"formats": ["json", "txt"], "model_revision": "fixture"}, state, calls, module


def test_content_duplicate_and_integrity(task_env, tmp_path):
    source, config, state, calls, _ = task_env
    first = run_task(source, config, tmp_path / "out")
    assert len(calls) == 1 and not first["duplicate"]
    second = run_task(source, config, tmp_path / "out")
    assert second["duplicate"] and second["outputs"] == first["outputs"]
    changed = Path(first["outputs"][0])
    changed.write_text("user edited result", encoding="utf-8")
    third = run_task(source, config, tmp_path / "out")
    assert len(calls) == 1  # re-export from verified transcript checkpoint
    assert third["outputs"] != first["outputs"]
    assert changed.read_text() == "user edited result"
    assert all(Path(p).is_file() for p in third["outputs"])
    event_text = "".join(p.read_text() for p in state.rglob("events.jsonl"))
    assert source.name not in event_text and "Fixture text" not in event_text


def test_renamed_input_and_new_output_root(task_env, tmp_path):
    source, config, _, calls, _ = task_env
    first = run_task(source, config, tmp_path / "out")
    copy = tmp_path / "renamed.wav"
    copy.write_bytes(source.read_bytes())
    assert fingerprint(source, config) == fingerprint(copy, config)
    assert fingerprint(source, config) != fingerprint(copy, dict(config, formats=["json"]))
    second = run_task(copy, config, tmp_path / "elsewhere")
    assert len(calls) == 1
    assert first["outputs"] != second["outputs"]
    assert all(str(tmp_path / "elsewhere") in p for p in second["outputs"])


def test_failure_retry_and_interruption(task_env, tmp_path):
    source, config, state, _, module = task_env

    def fail(*_):
        raise RuntimeError("private transcript SECRET and private-source.wav")

    module.transcribe = fail
    with pytest.raises(RuntimeError):
        run_task(source, config, tmp_path / "out")
    with pytest.raises(RetryRequiredError):
        run_task(source, config, tmp_path / "out")
    status = next(state.rglob("job.json"))
    assert json.loads(status.read_text())["status"] == "failed"
    assert "SECRET" not in next(state.rglob("events.jsonl")).read_text()

    def interrupt(*_):
        raise KeyboardInterrupt()

    module.transcribe = interrupt
    with pytest.raises(KeyboardInterrupt):
        run_task(source, config, tmp_path / "out", retry=True)
    assert json.loads(status.read_text())["status"] == "interrupted"
    assert not list(state.rglob("task.lock"))


def test_lock_live_and_stale(tmp_path, monkeypatch):
    lock = tmp_path / "task.lock"
    with task_lock(lock):
        with pytest.raises(TaskBusyError):
            with task_lock(lock):
                pass
    assert not lock.exists()
    lock.write_text("999999999")
    original_kill = os.kill

    def fake_kill(pid, sig):
        if pid == 999999999:
            raise ProcessLookupError()
        return original_kill(pid, sig)

    monkeypatch.setattr(os, "kill", fake_kill)
    with task_lock(lock):
        assert lock.read_text() == str(os.getpid())
    assert not lock.exists()


def test_stream_digest(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"x" * (2 * 1024 * 1024 + 10))
    assert len(digest(path)) == 64


def test_source_changed_during_inference_rejected(task_env, tmp_path):
    source, config, state, _, module = task_env

    def mutate(source, config, job_dir):
        source.write_bytes(b"different input now")
        return {
            "metadata": {
                "source_file": "fixture.wav",
                "duration_ms": 0,
                "language": "zh",
                "model": "fixture",
            },
            "speakers": [],
            "segments": [],
        }

    module.transcribe = mutate
    with pytest.raises(ValueError, match="Source file changed"):
        run_task(source, config, tmp_path / "out")
    assert not list((tmp_path / "out").rglob("transcript.*"))
    assert json.loads(next(state.rglob("job.json")).read_text())["status"] == "failed"


def test_corrupt_transcript_checkpoint_triggers_pipeline(task_env, tmp_path):
    source, config, state, calls, _ = task_env
    first = run_task(source, config, tmp_path / "out")
    cached = next(state.rglob("transcript.json"))
    cached.write_text("corrupted checkpoint", encoding="utf-8")
    Path(first["outputs"][0]).write_text("edited export", encoding="utf-8")
    run_task(source, config, tmp_path / "out")
    assert len(calls) == 2
    assert json.loads(cached.read_text())["metadata"]["model"] == "fixture-only"


def test_progress_private_bounded_and_no_events(tmp_path):
    from local_media_transcriber.task_manager.manager import atomic_json, record_progress

    atomic_json(
        tmp_path / "job.json",
        {"task_id": "test-task", "status": "running", "progress": 0, "error_code": None},
    )
    record_progress(tmp_path, 2, 4)
    record = json.loads((tmp_path / "job.json").read_text())
    assert record == {
        "task_id": "test-task",
        "status": "running",
        "progress": 0.45,
        "error_code": None,
    }
    assert not (tmp_path / "events.jsonl").exists()
    record_progress(tmp_path, 10, 4)
    assert json.loads((tmp_path / "job.json").read_text())["progress"] == 0.9


def test_damaged_private_job_state_recovers_without_overwrite(task_env, tmp_path):
    source, config, state, calls, _ = task_env
    first = run_task(source, config, tmp_path / "out")
    next(state.rglob("job.json")).write_text("invalid", encoding="utf-8")
    next(state.rglob("artifacts.json")).write_text("[]", encoding="utf-8")
    second = run_task(source, config, tmp_path / "out")
    assert len(calls) == 2
    assert second["outputs"] != first["outputs"]
    assert all(Path(p).is_file() for p in first["outputs"])
