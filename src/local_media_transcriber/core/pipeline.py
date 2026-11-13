"""Local VAD -> raw ASR -> punctuation -> global speaker clustering -> postprocessing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Real
import re
import time
import tempfile
import uuid
from pathlib import Path

from .runtime import offline_runtime, quiet_runtime
from ..audio.media import probe, extract_chunk, analyze_channels
from ..models.registry import get_model, model_path, select_model
from ..postprocessing import process_segments
from ..diarization.clustering import assign_speakers


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if k != "spk_embedding"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(type(value).__name__)


class Engine:
    def __init__(self, spec, config):
        import os
        from .paths import cache_dir

        for key, child in (
            ("HF_HOME", "hub"),
            ("MODELSCOPE_CACHE", "modelscope"),
            ("TORCH_HOME", "torch"),
            ("NUMBA_CACHE_DIR", "jit"),
        ):
            os.environ.setdefault(key, str(cache_dir() / child))
        from .compat import install_audio_only_adapter

        install_audio_only_adapter()
        from funasr import AutoModel
        import torch

        self.spec, self.config = spec, config
        threads = int(config.get("threads", 4))
        torch.set_num_threads(threads)
        base = dict(
            device="cpu",
            ncpu=threads,
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
            trust_remote_code=False,
            log_level="ERROR",
        )
        self.asr = AutoModel(model=str(model_path(spec["name"])), **base)
        vad_spec = get_model("fsmn-vad")
        if vad_spec["sample_rate"] != spec["sample_rate"]:
            raise ValueError(
                "Selected ASR and VAD sample rates differ; add a compatible VAD adapter"
            )
        self.vad = AutoModel(model=str(model_path("fsmn-vad")), **base)
        self.punc = None
        # CT-Punc is separate from raw ASR and only supports zh/en.
        if spec["name"] == "paraformer-zh":
            self.punc = AutoModel(model=str(model_path("ct-punc")), **base)
        self.spk = None
        if config.get("diarization"):
            if get_model("cam++")["sample_rate"] != spec["sample_rate"]:
                raise ValueError("CAM++ requires its configured sample rate")
            self.spk = AutoModel(model=str(model_path("cam++")), **base)

    def chunk(self, file: Path, offset: int, channel: int | None):
        import soundfile as sf

        waveform, sr = sf.read(file, dtype="float32")
        duration = round(len(waveform) / sr * 1000)
        vad = self.vad.generate(input=waveform, fs=sr, cache={}, max_single_segment_time=30000)
        ranges = vad[0].get("value", []) if vad else []
        records = []
        for left, right in ranges:
            left, right = max(0, int(left)), min(duration, int(right))
            if right <= left:
                continue
            # Bound any VAD interval even if an upstream version ignores its length option.
            for begin in range(left, right, 30000):
                end = min(begin + 30000, right)
                audio = waveform[round(begin * sr / 1000) : round(end * sr / 1000)]
                kwargs = dict(input=audio, fs=sr, cache={}, batch_size=1)
                if self.spec["name"] == "sensevoice-small":
                    kwargs.update(language=self.config.get("language", "auto"), use_itn=True)
                if self.spec["name"] == "paraformer-zh":
                    kwargs["output_timestamp"] = True
                output = self.asr.generate(**kwargs)
                for result in output:
                    raw_text = result.get("text", "")
                    if not raw_text.strip():
                        continue
                    text = re.sub(r"<\|[^|]+\|>", "", raw_text).strip()
                    detected = re.search(r"<\|(zh|en|yue|ja|ko)\|>", raw_text)
                    punctuation = None
                    # SenseVoice supplies ITN punctuation in its raw output. Do not
                    # run a second punctuation model over already punctuated text.
                    if self.punc and text:
                        punctuation = self.punc.generate(input=text, cache={})
                    speaker_windows = []
                    raw_alignment = result.get("timestamp")
                    aligned_intervals = None
                    if isinstance(raw_alignment, list):
                        aligned_intervals = [
                            [offset + begin + t[0], offset + begin + t[1]]
                            for t in raw_alignment
                            if isinstance(t, (list, tuple))
                            and len(t) >= 2
                            and all(isinstance(v, Real) and math.isfinite(v) for v in t[:2])
                        ]
                    if self.spk:
                        # VAD intervals can contain speaker changes. Extract short acoustic
                        # windows and cluster globally; never equate a channel with a person.
                        for position in range(0, len(audio), round(sr * 0.75)):
                            window = audio[position : position + round(sr * 1.5)]
                            if len(window) < sr // 2:
                                continue
                            spk = self.spk.generate(input=window, fs=sr, cache={})
                            if spk and "spk_embedding" in spk[0]:
                                from ..audio.quality import window_evidence

                                window_start = offset + begin + round(position / sr * 1000)
                                speaker_windows.append(
                                    {
                                        "start_ms": offset + begin + round(position / sr * 1000),
                                        "end_ms": offset
                                        + begin
                                        + round((position + len(window)) / sr * 1000),
                                        "speaker_id": None,
                                        "channel": channel,
                                        **window_evidence(
                                            window, sr, window_start, aligned_intervals
                                        ),
                                        "embedding": spk[0]["spk_embedding"]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                        .mean(axis=0)
                                        .tolist(),
                                    }
                                )
                    records.append(
                        {
                            "start_ms": offset + begin,
                            "end_ms": offset + end,
                            "speaker_id": None,
                            "text": text,
                            "asr_raw_text": raw_text,
                            "channel": channel,
                            "timing_precision": "vad_interval",
                            "raw_result": _jsonable(result),
                            "punctuation": _jsonable(punctuation),
                            "speaker_windows": speaker_windows,
                            "detected_language": detected.group(1) if detected else None,
                        }
                    )
        return records


def _checkpoint_write(path, value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    envelope = {"sha256": hashlib.sha256(data).hexdigest(), "payload": value}
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(envelope, ensure_ascii=False))
    temp.replace(path)


def _checkpoint_read(path):
    try:
        value = json.loads(path.read_text())
        data = json.dumps(value["payload"], ensure_ascii=False, allow_nan=False).encode()
        return value["payload"] if hashlib.sha256(data).hexdigest() == value["sha256"] else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _sentences(record):
    """Use model token alignment when supported; otherwise retain measured VAD interval."""
    raw = record["raw_result"]
    punctuation = record.get("punctuation")
    base = {
        k: record[k]
        for k in ("start_ms", "end_ms", "speaker_id", "text", "channel", "timing_precision")
    }
    if not punctuation:
        return [base]
    punc = punctuation[0]
    text = punc.get("text", base["text"])
    timestamps = raw.get("timestamp")
    if timestamps and punc.get("punc_array"):
        try:
            from funasr.utils.timestamp_tools import timestamp_sentence

            sentences = timestamp_sentence(
                punc["punc_array"], timestamps, raw["text"], return_raw_text=True
            )
            if sentences:
                output = []
                for sentence in sentences:
                    start, end = sentence.get("start"), sentence.get("end")
                    if start is None or end is None:
                        raise ValueError("No alignment")
                    # Token timestamps are relative to this measured VAD interval.
                    if (
                        start < 0
                        or end < start
                        or start > base["end_ms"] - base["start_ms"]
                        or end > base["end_ms"] - base["start_ms"] + 100
                    ):
                        raise ValueError("Invalid alignment")
                    output.append(
                        {
                            **base,
                            "start_ms": base["start_ms"] + int(start),
                            "end_ms": min(base["end_ms"], base["start_ms"] + int(end)),
                            "text": sentence.get("text", text),
                            "timing_precision": "model_token_alignment",
                        }
                    )
                return output
        except (ValueError, TypeError, KeyError, IndexError, AssertionError):
            pass
    return [{**base, "text": text}]


def _runtime_versions():
    from importlib.metadata import version, PackageNotFoundError

    versions = {}
    for package in ("funasr", "torch", "torchaudio", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def transcribe(source: Path, config: dict, job_dir: Path) -> dict:
    start = time.perf_counter()
    info = probe(source)
    spec = select_model(config.get("language", "zh"), config.get("model"))
    sample_rate = spec["sample_rate"]
    chunk_seconds = float(config.get("chunk_seconds", 60))
    if not math.isfinite(chunk_seconds) or not 1 <= chunk_seconds <= 600:
        raise ValueError("chunk_seconds must be between 1 and 600")
    chunk_ms = round(chunk_seconds * 1000)
    analysis = analyze_channels(source)
    channel_policy = str(config.get("channel", "split"))
    channels = (
        list(range(info["channels"]))
        if channel_policy == "split"
        else [None if channel_policy == "mix" else int(channel_policy)]
    )
    if any(c is not None and (c < 0 or c >= info["channels"]) for c in channels):
        raise ValueError("Requested audio channel is unavailable")
    job_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    checkpoint_dir = job_dir / "chunks"
    invalid_marker = job_dir / "input-integrity-invalid.json"
    if invalid_marker.exists():
        if checkpoint_dir.exists():
            checkpoint_dir.rename(job_dir / ("invalid-chunks-" + uuid.uuid4().hex))
        invalid_marker.unlink()
    checkpoint_dir.mkdir(exist_ok=True, mode=0o700)
    records, resumed_chunks, engine = [], 0, None
    completed_chunks = 0
    total_chunks = len(channels) * math.ceil(info["duration_ms"] / chunk_ms)
    from ..task_manager.manager import record_progress

    # Loading models and all upstream operations happen under an offline/quiet boundary.
    with offline_runtime(), quiet_runtime():
        for channel in channels:
            for offset in range(0, info["duration_ms"], chunk_ms):
                file = checkpoint_dir / f"{channel}-{offset}.json"
                payload = _checkpoint_read(file) if file.exists() else None
                if payload is not None:
                    records.extend(payload)
                    resumed_chunks += 1
                    completed_chunks += 1
                    record_progress(job_dir, completed_chunks, total_chunks)
                    continue
                if engine is None:
                    engine = Engine(spec, config)
                with tempfile.TemporaryDirectory(prefix="lmt-", dir=job_dir) as temp:
                    wav = Path(temp) / "chunk.wav"
                    extract_chunk(
                        source,
                        wav,
                        offset,
                        min(offset + chunk_ms, info["duration_ms"]),
                        sample_rate,
                        channel,
                    )
                    payload = engine.chunk(wav, offset, channel)
                _checkpoint_write(file, payload)
                records.extend(payload)
                completed_chunks += 1
                record_progress(job_dir, completed_chunks, total_chunks)
    records.sort(key=lambda r: (r["start_ms"], r["channel"] if r["channel"] is not None else -1))
    warnings = [
        "ASR output requires human review; confidence and identity are not verified.",
        "Fixed chunk boundaries can split words; sentence times may be coarse VAD intervals.",
        "Only the first audio stream is transcribed; original stream metadata is retained.",
    ]
    if info["channels"] > 1:
        warnings.append(
            "Channels are not speakers. Split mode may duplicate shared stereo content; use --channel mix after inspection if appropriate."
        )
    speakers = []
    if config.get("diarization"):
        with quiet_runtime():
            windows = [w for record in records for w in record.get("speaker_windows", [])]
            speakers, extra = assign_speakers(
                windows, [w["embedding"] for w in windows], config.get("speakers")
            )
            for record in records:
                labels = {
                    w["speaker_id"]
                    for w in record.get("speaker_windows", [])
                    if w["speaker_id"] is not None
                }
                record["speaker_id"] = next(iter(labels)) if len(labels) == 1 else None
        warnings.extend(extra)
    for speaker in speakers:
        speaker["display_name"] = config.get("speaker_map", {}).get(speaker["id"], speaker["id"])
    raw_segments, processed, changes = [], [], []
    detected = set()
    for record in records:
        if record.get("detected_language"):
            detected.add(record["detected_language"])
        raw_segments.append(
            {
                k: record[k]
                for k in ("start_ms", "end_ms", "speaker_id", "text", "channel", "timing_precision")
            }
        )
        raw_segments[-1]["text"] = record["asr_raw_text"]
        for sentence in _sentences(record):
            if config.get("diarization"):
                votes = {}
                for window in record.get("speaker_windows", []):
                    overlap = max(
                        0,
                        min(sentence["end_ms"], window["end_ms"])
                        - max(sentence["start_ms"], window["start_ms"]),
                    )
                    label = window["speaker_id"]
                    if label is not None and overlap:
                        votes[label] = votes.get(label, 0) + overlap
                sentence["speaker_id"] = max(votes, key=votes.get) if votes else None
                sentence["speaker_uncertain"] = (
                    len([v for v in votes.values() if v >= sum(votes.values()) * 0.2]) > 1
                )
            sentence["id"] = len(processed) + 1
            sentence.update(
                translation=None,
                translation_language=None,
                paragraph_id=len(processed) + 1,
                audio_position_ms=sentence["start_ms"],
            )
            processed.append(sentence)
        if record.get("punctuation") or record["asr_raw_text"] != record["text"]:
            changes.append(
                {
                    "raw_segment_id": len(raw_segments),
                    "field": "text",
                    "before": record["asr_raw_text"],
                    "after": " ".join(s["text"] for s in _sentences(record)),
                    "reason": "ct_punc_or_rich_tag_removal_and_alignment",
                }
            )
    for index, raw in enumerate(raw_segments, 1):
        raw["id"] = index
    processed, edits = process_segments(
        processed,
        config.get("glossary"),
        config.get("reading", False),
        config.get("time_offset_ms", 0),
    )
    changes.extend(edits)
    for speaker in speakers:
        matched = [seg for seg in processed if seg["speaker_id"] == speaker["id"]]
        speaker["segment_count"] = len(matched)
        speaker["duration_ms"] = sum(seg["end_ms"] - seg["start_ms"] for seg in matched)
        if speaker["display_name"] != speaker["id"]:
            changes.append(
                {
                    "speaker_id": speaker["id"],
                    "field": "display_name",
                    "before": speaker["id"],
                    "after": speaker["display_name"],
                    "reason": "user_speaker_map",
                }
            )
    if any(segment.get("speaker_uncertain") for segment in processed):
        warnings.append(
            "Some sentences span conflicting short-window speaker labels; marked speaker_uncertain for manual review."
        )
    language = config.get("language", "zh")
    if language == "auto":
        language = (
            next(iter(detected))
            if len(detected) == 1
            else ("mixed" if detected else "undetermined")
        )
    elapsed = time.perf_counter() - start
    peak_rss = None
    try:
        import resource
        import sys

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
    except ImportError:
        pass
    result = {
        "metadata": {
            "source_file": source.name,
            **info,
            "language": language,
            "requested_language": config.get("language", "zh"),
            "model": spec["name"],
            "model_revision": spec["version"],
            "device": "cpu",
            "diarization": bool(config.get("diarization")),
            "diarization_method": "campp_1.5s_windows_0.75s_hop_global_clustering"
            if config.get("diarization")
            else None,
            "speaker_count_method": (
                "explicit_constraint"
                if config.get("speakers") is not None
                else "cosine_knn_multiscale_short_evidence_v2"
            )
            if config.get("diarization")
            else None,
            "estimated_speaker_count": len(speakers) if config.get("diarization") else None,
            "speaker_count_is_verified": False,
            "mode": config.get("mode", "transcription"),
            "channel_policy": channel_policy,
            "channel_analysis": analysis,
            "elapsed_seconds": round(elapsed, 3),
            "peak_rss_bytes": peak_rss,
            "real_time_factor": round(elapsed / max(info["duration_ms"] / 1000, 0.001), 3),
            "resumed_chunks": resumed_chunks,
            "sample_rate": sample_rate,
            "original_sample_rate": info["sample_rate"],
            "runtime_versions": _runtime_versions(),
        },
        "speakers": speakers,
        "segments": processed,
        "raw_transcript": {
            "segments": raw_segments,
            "model_results": [r["raw_result"] for r in records],
        },
        "processed_transcript": {"segments": copy.deepcopy(processed)},
        "changes": changes,
        "warnings": warnings,
    }
    if config.get("mode") == "learning":
        from ..postprocessing.learning import learning_sentences

        result["learning_sentences"] = learning_sentences(processed)
        if any(
            s["timing_precision"] == "shared_source_interval" for s in result["learning_sentences"]
        ):
            warnings.append(
                "Learning sentences share a source interval where fine alignment is unavailable; their exact boundaries are unknown."
            )
    if config.get("export_exercises"):
        result["exercises"] = [
            {
                "segment_id": s.get("segment_id", s.get("id")),
                "sentence_id": s.get("sentence_id"),
                "timing_precision": s.get("timing_precision"),
                "start_ms": s["start_ms"],
                "end_ms": s["end_ms"],
                "prompt": "Listen and transcribe; review the machine reference.",
                "machine_reference": s["text"],
                "words": re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", s["text"]),
                "human_verified": False,
            }
            for s in result.get("learning_sentences", processed)
        ]
    return result
