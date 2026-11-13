#!/usr/bin/env python3
"""Real local automatic-count regression and held-out synthetic-voice evaluation.

Use a new private --work-dir. No models are downloaded and no ASR is mocked.
Generated media, transcripts and acoustic embeddings must stay out of Git.
"""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import wave


def evaluate_speaker_count(
    transcript: dict, references: list[dict], minimum_agreement: float = 0.8
) -> dict:
    """Score exact count and balanced per-voice sentence labels up to permutation.

    Sentence midpoints identify the reference interval. Voices without any
    matching transcript sentence score zero, so a long voice cannot hide a
    completely missed short third voice. This is not a DER implementation.
    """
    voices = list(dict.fromkeys(item["voice"] for item in references))
    if not voices or len(voices) > 3:
        raise ValueError("Evaluation requires one to three reference voices")
    if not 0 <= minimum_agreement <= 1:
        raise ValueError("Agreement threshold must be between zero and one")
    votes = {voice: Counter() for voice in voices}
    totals = Counter()
    for segment in transcript["segments"]:
        left, right = segment.get("start_ms"), segment.get("end_ms")
        if left is None or right is None:
            continue
        midpoint = (left + right) / 2
        reference = next(
            (item for item in references if item["start_ms"] <= midpoint < item["end_ms"]), None
        )
        if reference is not None:
            voice = reference["voice"]
            totals[voice] += 1
            if segment.get("speaker_id") is not None:
                votes[voice][segment["speaker_id"]] += 1
    labels = sorted({label for counts in votes.values() for label in counts}, key=str)
    candidates = labels + [None] * max(0, len(voices) - len(labels))
    best_scores = {voice: 0.0 for voice in voices}
    best_total = -1.0
    for assignment in itertools.permutations(candidates, len(voices)):
        scores = {
            voice: (
                votes[voice][label] / totals[voice] if label is not None and totals[voice] else 0.0
            )
            for voice, label in zip(voices, assignment)
        }
        if sum(scores.values()) > best_total:
            best_scores, best_total = scores, sum(scores.values())
    expected, observed = len(voices), len(transcript["speakers"])
    macro, minimum = sum(best_scores.values()) / expected, min(best_scores.values())
    failures = []
    if expected != observed:
        failures.append("count_mismatch")
    if minimum < minimum_agreement:
        failures.append("per_voice_agreement_below_threshold")
    return {
        "expected_count": expected,
        "observed_count": observed,
        "count_match": expected == observed,
        "per_voice_agreement": best_scores,
        "reference_sentence_counts": {voice: totals[voice] for voice in voices},
        "macro_agreement": macro,
        "min_agreement": minimum,
        "required_min_agreement": minimum_agreement,
        "status": "failed" if failures else "passed",
        "failure_reasons": failures,
    }


def concat_entry(path: Path) -> str:
    """FFmpeg concat quoting, without routing a filename through the shell."""
    value = str(path.resolve())
    if "\n" in value or "\r" in value:
        raise ValueError("FFmpeg concat paths cannot contain line breaks")
    return "file '" + value.replace("'", "'\\''") + "'\n"


def _ff(*arguments) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-n", *map(str, arguments)],
        check=True,
        capture_output=True,
    )


def _duration(path: Path) -> int:
    with wave.open(str(path)) as file:
        return round(file.getnframes() / file.getframerate() * 1000)


def generate_cases(root: Path, original_fixtures: Path | None = None) -> list[dict]:
    clips = {}

    def say(key, voice, text, rate=160, role="A"):
        aiff, wav = root / f"{key}.aiff", root / f"{key}.wav"
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text],
            check=True,
            capture_output=True,
        )
        _ff("-i", aiff, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav)
        clips[key] = (wav, role)

    original = [
        (
            "oa",
            "zh.wav",
            "Tingting",
            "今天我们测试本地音视频转录工具。请保留原始文件，检查时间戳，并且确认全部内容都在电脑上处理。",
            "A",
        ),
        (
            "ob",
            "zh_male.wav",
            "Eddy (中文（中国大陆）)",
            "好的，我来核对输出结果。我们需要逐字稿和字幕文件，也需要检查两位说话人的编号。",
            "B",
        ),
        (
            "oc",
            "zh_third.wav",
            "Flo (中文（中国大陆）)",
            "我补充第三个观点。请保存完整逐字稿，并为每个发言标注时间戳。",
            "C",
        ),
    ]
    for key, filename, voice, text, role in original:
        if original_fixtures:
            source = original_fixtures / filename
            if not source.is_file():
                raise FileNotFoundError("Original fixture audio is missing")
            destination = root / f"{key}.wav"
            shutil.copyfile(source, destination)
            clips[key] = (destination, role)
        else:
            say(key, voice, text, role=role)
    new = [
        (
            "solo_slow",
            "Tingting",
            "清晨的图书馆还很安静。我先把昨天记录的想法重新读一遍，再检查今天需要处理的事项。",
            125,
            "A",
        ),
        (
            "solo_fast",
            "Tingting",
            "下午我们准备去公园散步，带上雨伞和水杯，回来以后再整理书桌上的资料。",
            205,
            "A",
        ),
        (
            "a1",
            "Tingting",
            "这次讨论的是周末的出行安排。我们可以先比较交通方式，再决定集合的时间。",
            145,
            "A",
        ),
        (
            "a2",
            "Tingting",
            "我已经看过路线了。沿途有几处适合休息的地方，大家不用担心行程太紧。",
            190,
            "A",
        ),
        (
            "b1",
            "Eddy (中文（中国大陆）)",
            "我建议先核对天气预报。如果当天有雨，我们就把室外活动改到下一周。",
            175,
            "B",
        ),
        (
            "b2",
            "Eddy (中文（中国大陆）)",
            "交通费用还需要再算一次。确认人数以后，我会把预算和车次发给大家。",
            145,
            "B",
        ),
        (
            "c1",
            "Meijia",
            "我可以负责准备午餐。请提前告诉我有没有需要避开的食物，这样比较方便安排。",
            155,
            "C",
        ),
        (
            "c2",
            "Meijia",
            "还有一个小提醒。我们最好带上充电器，并且保留一份离线地图，避免临时找不到路。",
            185,
            "C",
        ),
        (
            "c_short_source",
            "Meijia",
            "这里还有一位同事正在补充重要的说明，请大家稍等一下。",
            165,
            "C",
        ),
    ]
    for key, voice, text, rate, role in new:
        say(key, voice, text, rate, role)
    for key, seconds in [("c_short", 3.0), ("c_tiny", 1.5)]:
        wav = root / f"{key}.wav"
        _ff("-i", clips["c_short_source"][0], "-t", str(seconds), "-c:a", "pcm_s16le", wav)
        clips[key] = (wav, "C")
    pause = root / "pause.wav"
    _ff("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "0.9", "-c:a", "pcm_s16le", pause)
    cases = []

    def build(name, keys, required=True):
        paths, references, cursor = [], [], 0
        for key in keys:
            if paths:
                paths.append(pause)
                cursor += _duration(pause)
            path, voice = clips[key]
            duration = _duration(path)
            references.append({"voice": voice, "start_ms": cursor, "end_ms": cursor + duration})
            paths.append(path)
            cursor += duration
        listing = root / f"{name}.ffconcat"
        listing.write_text("".join(concat_entry(path) for path in paths), encoding="utf-8")
        destination = root / f"{name}.wav"
        _ff("-f", "concat", "-safe", "0", "-i", listing, "-c:a", "pcm_s16le", destination)
        cases.append(
            {"case": name, "source": destination, "references": references, "required": required}
        )

    build("original_single", ["oa"])
    build("original_two", ["oa", "ob", "oa", "ob"])
    build("original_three", ["oa", "ob", "oc", "oa", "ob", "oc"])
    build("heldout_single_changed_text_rate", ["solo_slow", "solo_fast"])
    build("heldout_two_changed_text_rate", ["a1", "b1", "a2", "b2"])
    build("heldout_three_new_voice", ["a1", "b1", "c1", "a2", "b2", "c2"])
    build("heldout_short_third_3s", ["a1", "b1", "c_short", "a2", "b2"])
    build("diagnostic_tiny_third_1_5s", ["a1", "b1", "c_tiny", "a2", "b2"], required=False)
    return cases


def legacy_count_from_checkpoints(job_dir: Path) -> dict:
    """Replay the old 0.35 rule over the same real, checksummed CAM++ windows."""
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from local_media_transcriber.core.pipeline import _checkpoint_read

    windows = []
    for path in sorted((job_dir / "chunks").glob("*.json")):
        records = _checkpoint_read(path)
        if records is None:
            raise ValueError("Invalid private checkpoint")
        windows.extend(window for record in records for window in record.get("speaker_windows", []))
    vectors = [
        np.asarray(window["embedding"], dtype="float32")
        for window in windows
        if window.get("embedding") is not None
    ]
    if not vectors:
        return {"legacy_count": 0, "window_count": 0}
    matrix = np.stack(vectors)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-10)
    if len(matrix) == 1:
        count = 1
    else:
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.35
        ).fit_predict(matrix)
        count = len(set(labels.tolist()))
    return {"legacy_count": count, "window_count": len(matrix)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--original-fixtures", type=Path, help="Optional v0.1.0 private fixture directory"
    )
    args = parser.parse_args(argv)
    root = args.work_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(root.iterdir()):
        parser.error("Use a new empty private work directory")
    if not shutil.which("say"):
        parser.error("Fixture generation requires macOS say")
    cases = generate_cases(
        root, args.original_fixtures.expanduser().resolve() if args.original_fixtures else None
    )
    from local_media_transcriber.cli import parser as cli_parser, _configuration
    from local_media_transcriber.core.paths import state_dir
    from local_media_transcriber.exporters import validate_transcript
    from local_media_transcriber.task_manager.manager import run_task
    from local_media_transcriber import __version__

    report = {
        "version": __version__,
        "data": "locally synthesized original text; no human-recording accuracy claim",
        "cases": [],
    }
    for case in cases:
        expected = len({reference["voice"] for reference in case["references"]})
        row = {
            "case": case["case"],
            "required": case["required"],
            "expected_count": expected,
            "observed_count": None,
            "count_match": False,
            "per_voice_agreement": {},
            "macro_agreement": 0.0,
            "min_agreement": 0.0,
            "status": "failed",
        }
        try:
            config = _configuration(
                cli_parser().parse_args(
                    [
                        "run",
                        str(case["source"]),
                        "--mode",
                        "diarization",
                        "--channel",
                        "mix",
                        "--formats",
                        "json",
                    ]
                )
            )
            result = run_task(case["source"], config, root / "results")
            transcript = json.loads(Path(result["outputs"][0]).read_text())
            validate_transcript(transcript)
            row.update(evaluate_speaker_count(transcript, case["references"]))
            row.update(legacy_count_from_checkpoints(state_dir() / "jobs" / result["task_id"]))
            row.update(
                metadata=transcript["metadata"],
                outputs=result["outputs"],
                warnings=transcript.get("warnings", []),
            )
        except Exception as error:
            row.update(status="failed", error_code=type(error).__name__)
        report["cases"].append(row)
        (root / "speaker-count-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    key: row.get(key)
                    for key in [
                        "case",
                        "required",
                        "status",
                        "expected_count",
                        "observed_count",
                        "legacy_count",
                        "count_match",
                        "macro_agreement",
                        "min_agreement",
                        "error_code",
                    ]
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    required_failures = [
        row["case"] for row in report["cases"] if row["required"] and row["status"] != "passed"
    ]
    diagnostic_failures = [
        row["case"] for row in report["cases"] if not row["required"] and row["status"] != "passed"
    ]
    report["required_failures"], report["diagnostic_failures"] = (
        required_failures,
        diagnostic_failures,
    )
    report["status"] = (
        "failed"
        if required_failures
        else ("passed_with_diagnostic_failures" if diagnostic_failures else "passed")
    )
    (root / "speaker-count-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return 1 if required_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
