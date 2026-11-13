#!/usr/bin/env python3
"""Reproducible synthetic-media integration checks; no private files or ASR mocks.

macOS: download the five registered models first, then run with a new --work-dir.
Results stay in that directory. Synthetic voices are not a real-world accuracy corpus.
"""

import argparse
import json
import shutil
import subprocess
import wave
import itertools
from pathlib import Path

from local_media_transcriber.cli import parser, _configuration
from local_media_transcriber.task_manager.manager import run_task
from local_media_transcriber.exporters import validate_transcript


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", *map(str, args)], check=True)


def config(*args):
    return _configuration(parser().parse_args(["run", "placeholder.wav", *args]))


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--work-dir", type=Path, required=True)
    args = arguments.parse_args()
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        arguments.error("Use an empty work directory")
    if not shutil.which("say"):
        arguments.error("Synthetic fixture generation currently requires macOS say")
    texts = {
        "zh_third": (
            "Flo (中文（中国大陆）)",
            "我补充第三个观点。请保存完整逐字稿，并为每个发言标注时间戳。",
        ),
        "zh": (
            "Tingting",
            "今天我们测试本地音视频转录工具。请保留原始文件，检查时间戳，并且确认全部内容都在电脑上处理。",
        ),
        "zh_male": (
            "Eddy (中文（中国大陆）)",
            "好的，我来核对输出结果。我们需要逐字稿和字幕文件，也需要检查两位说话人的编号。",
        ),
        "en": (
            "Samantha",
            "Today we are testing local audio transcription. Please keep the original recording and check the timestamps. No audio is uploaded to a cloud service.",
        ),
    }
    for name, (voice, text) in texts.items():
        subprocess.run(
            ["say", "-v", voice, "-r", "160", "-o", str(root / f"{name}.aiff"), text], check=True
        )
        ff("-i", root / f"{name}.aiff", "-ar", "16000", "-ac", "1", root / f"{name}.wav")
    ff("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "0.9", root / "pause.wav")

    def concat(names, destination):
        listing = root / f"{destination}.list"
        listing.write_text("".join(f"file '{root / (n + '.wav')}'\n" for n in names))
        ff(
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            listing,
            "-c:a",
            "pcm_s16le",
            root / f"{destination}.wav",
        )

    concat(["zh", "pause", "zh_male", "pause", "zh", "pause", "zh_male"], "two")
    concat(
        [
            "zh",
            "pause",
            "zh_male",
            "pause",
            "zh_third",
            "pause",
            "zh",
            "pause",
            "zh_male",
            "pause",
            "zh_third",
        ],
        "three",
    )
    concat(["zh", "pause", "en"], "mixed")
    concat(["two", "pause", "two"], "long")
    ff("-i", root / "zh.wav", "-c:a", "libmp3lame", root / "zh.mp3")
    ff("-i", root / "zh.wav", "-c:a", "aac", root / "zh.m4a")
    ff(
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x180:r=10",
        "-i",
        root / "zh.wav",
        "-c:v",
        "mpeg4",
        "-c:a",
        "aac",
        "-shortest",
        root / "zh.mp4",
    )
    ff("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "5", root / "silence.wav")
    ff(
        "-f",
        "lavfi",
        "-i",
        "anoisesrc=color=white:amplitude=0.1:sample_rate=16000:seed=17",
        "-t",
        "5",
        root / "noise.wav",
    )
    ff(
        "-i",
        root / "zh.wav",
        "-i",
        root / "zh_male.wav",
        "-filter_complex",
        "[0:a][1:a]amerge=inputs=2",
        root / "stereo.wav",
    )
    cases = [
        (
            "chinese_wav",
            "zh.wav",
            config("--formats", "txt,srt,vtt,json,csv,md,docx", "--channel", "mix"),
        ),
        ("chinese_mp3", "zh.mp3", config("--channel", "mix")),
        ("chinese_m4a", "zh.m4a", config("--channel", "mix")),
        ("video_subtitles", "zh.mp4", config("--mode", "subtitle", "--channel", "mix")),
        (
            "english",
            "en.wav",
            config("--language", "en", "--mode", "learning", "--exercises", "--channel", "mix"),
        ),
        ("mixed_languages", "mixed.wav", config("--language", "zh", "--channel", "mix")),
        (
            "two_speakers",
            "two.wav",
            config("--mode", "diarization", "--speakers", "2", "--channel", "mix"),
        ),
        ("auto_speakers", "two.wav", config("--mode", "diarization", "--channel", "mix")),
        ("silence", "silence.wav", config("--mode", "subtitle", "--channel", "mix")),
        ("noise", "noise.wav", config("--channel", "mix")),
        ("separate_channels", "stereo.wav", config("--channel", "split")),
        ("long_chunked", "long.wav", config("--chunk-seconds", "20", "--channel", "mix")),
    ]
    cases.extend(
        [
            ("auto_language", "en.wav", config("--language", "auto", "--channel", "mix")),
            ("single_speaker", "zh.wav", config("--speakers", "1", "--channel", "mix")),
            (
                "three_speakers",
                "three.wav",
                config("--mode", "diarization", "--speakers", "3", "--channel", "mix"),
            ),
        ]
    )
    mapped = config("--mode", "research", "--speakers", "2", "--channel", "mix")
    mapped["speaker_map"] = {"SPEAKER_00": "受访者甲", "SPEAKER_01": "受访者乙"}
    cases.append(("speaker_mapping", "two.wav", mapped))
    reports = []

    def run(name, filename, options):
        try:
            result = run_task(root / filename, options, root / "results")
            from local_media_transcriber.core.paths import state_dir

            data = json.loads(
                (state_dir() / "jobs" / result["task_id"] / "transcript.json").read_text()
            )
            validate_transcript(data)
            report = {
                "case": name,
                "status": "passed",
                "metadata": data["metadata"],
                "segments": len(data["segments"]),
                "speakers": len(data["speakers"]),
                "speaker_counts": {s["id"]: s["segment_count"] for s in data["speakers"]},
                "outputs": result["outputs"],
                "duplicate": result["duplicate"],
                "timing_precision": sorted({s["timing_precision"] for s in data["segments"]}),
            }
            if name not in {"silence", "noise"} and not data["segments"]:
                raise AssertionError("Expected speech produced no segments")
            if name == "silence" and data["segments"]:
                raise AssertionError("Silence produced a hallucinated transcript")
            if name == "speaker_mapping" and {sp["display_name"] for sp in data["speakers"]} != {
                "受访者甲",
                "受访者乙",
            }:
                raise AssertionError("Human mapping failed")
            if name == "three_speakers" and len(data["speakers"]) != 3:
                raise AssertionError("Three-speaker count constraint failed")
            if name == "english" and not data.get("exercises"):
                raise AssertionError("Exercise export absent")
            if name == "two_speakers":
                if len(data["speakers"]) != 2:
                    raise AssertionError("Two-speaker clustering failed")
                timeline = []
                cursor = 0
                for voice in ["zh", "pause", "zh_male", "pause", "zh", "pause", "zh_male"]:
                    with wave.open(str(root / f"{voice}.wav")) as audio:
                        duration = audio.getnframes() / audio.getframerate() * 1000
                    if voice != "pause":
                        timeline.append((cursor, cursor + duration, voice))
                    cursor += duration
                pairs = []
                for segment in data["segments"]:
                    midpoint = (segment["start_ms"] + segment["end_ms"]) / 2
                    reference = next(
                        (voice for left, right, voice in timeline if left <= midpoint <= right),
                        None,
                    )
                    if reference:
                        pairs.append((reference, segment["speaker_id"]))
                labels = [speaker["id"] for speaker in data["speakers"]]
                best = max(
                    sum(mapping[voice] == label for voice, label in pairs)
                    for assignment in itertools.permutations(labels)
                    for mapping in [dict(zip(["zh", "zh_male"], assignment))]
                )
                report["synthetic_voice_label_agreement"] = best / len(pairs) if pairs else 0
                if report["synthetic_voice_label_agreement"] < 0.8:
                    raise AssertionError("Synthetic voice labels were inconsistent")
        except Exception as error:
            report = {"case": name, "status": "failed", "error_code": type(error).__name__}
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        (root / "validation-report.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2)
        )

    for case in cases:
        run(*case)
    run("duplicate_integrity", "zh.wav", cases[0][2])
    # Real ASR checkpoint interruption, injected only after a chunk has been committed.
    import local_media_transcriber.task_manager.manager as manager

    original = manager.record_progress

    def interrupt(job, completed, total):
        original(job, completed, total)
        if completed == 1:
            raise KeyboardInterrupt

    resume_config = config("--chunk-seconds", "10", "--channel", "mix")
    manager.record_progress = interrupt
    try:
        run_task(root / "long.wav", resume_config, root / "results")
    except KeyboardInterrupt:
        pass
    finally:
        manager.record_progress = original
    run("resume_after_interruption", "long.wav", resume_config)
    batch = root / "batch"
    batch.mkdir()
    shutil.copyfile(root / "zh.wav", batch / "chinese.wav")
    shutil.copyfile(root / "en.wav", batch / "english.wav")
    manifest = root / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "chinese.wav": {"mode": "transcription"},
                "english.wav": {"mode": "learning", "language": "en"},
            }
        )
    )
    from local_media_transcriber.cli import main as cli_main

    code = cli_main(
        [
            "batch",
            str(batch),
            "--recursive",
            "--manifest",
            str(manifest),
            "--output",
            str(root / "batch-results"),
            "--formats",
            "json",
        ]
    )
    reports.append({"case": "batch_manifest", "status": "passed" if code == 0 else "failed"})
    (root / "validation-report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0 if all(r["status"] == "passed" for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
