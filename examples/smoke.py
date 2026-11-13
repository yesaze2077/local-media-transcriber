#!/usr/bin/env python3
"""Real CPU inference smoke test. macOS say or --input provides lawful local audio."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Local test media on systems without macOS say")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source = args.input
    if source is None:
        if not shutil.which("say"):
            parser.error("Pass --input with a lawful local Chinese audio file on this OS")
        aiff = args.work_dir / "smoke.aiff"
        if aiff.exists():
            parser.error("Use a new empty work directory")
        subprocess.run(
            [
                "say",
                "-v",
                "Tingting",
                "-r",
                "160",
                "-o",
                str(aiff),
                "今天我们测试本地语音转录工具。所有音频都在电脑上处理，不会上传。",
            ],
            check=True,
        )
        source = args.work_dir / "smoke.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-n", "-i", str(aiff), "-ar", "16000", str(source)],
            check=True,
        )
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "transcribe"),
            "run",
            str(source),
            "--output",
            str(args.work_dir / "results"),
            "--formats",
            "json,srt",
            "--channel",
            "mix",
        ],
        env={**os.environ, "LMT_STATE_DIR": str(args.work_dir / "state")},
        capture_output=True,
        text=True,
    )
    if result.returncode:
        print(result.stderr)
        return result.returncode
    report = json.loads(result.stdout)
    output = next(Path(p) for p in report["outputs"] if p.endswith(".json"))
    transcript = json.loads(output.read_text())
    if not transcript["segments"] or not any(s["text"].strip() for s in transcript["segments"]):
        raise RuntimeError("CPU inference produced no text")
    print(
        json.dumps(
            {
                "status": "real_cpu_inference_passed",
                "model": transcript["metadata"]["model"],
                "duration_ms": transcript["metadata"]["duration_ms"],
                "segments": len(transcript["segments"]),
                "outputs": report["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
