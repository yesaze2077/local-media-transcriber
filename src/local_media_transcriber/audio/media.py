"""Bounded, local-only audio preprocessing through FFmpeg.

The source is never written. Every extraction explicitly chooses either a
single input channel or FFmpeg's mixdown. The caller decides that policy after
inspecting ``probe`` and optionally ``analyze_channels``.
"""

from __future__ import annotations

import array
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset(
    {
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
)


class MediaError(RuntimeError):
    """A local media inspection or conversion failed."""


def _tool(name: str) -> str:
    value = shutil.which(name)
    if value is None:
        raise MediaError(f"{name} is required; install FFmpeg and retry")
    return value


def _run(arguments: list[str], *, timeout: int = 300) -> bytes:
    try:
        result = subprocess.run(arguments, capture_output=True, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MediaError(f"{Path(arguments[0]).name} could not complete") from error
    if result.returncode:
        # FFmpeg diagnostics can contain a private filename or media tags.
        raise MediaError(
            f"{Path(arguments[0]).name} failed (exit {result.returncode}); check media and codecs"
        )
    return result.stdout


def probe(path: Path) -> dict:
    """Inspect the first audio stream, preserving original duration/channels.

    Metadata excludes arbitrary tags because they can carry personal data.
    A container without a readable audio stream is rejected.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise MediaError("Input is not a regular file")
    payload = _run(
        [
            _tool("ffprobe"),
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=30,
    )
    try:
        data = json.loads(payload)
        streams = [
            stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"
        ]
        if not streams:
            raise MediaError("No readable audio stream found")
        stream = streams[0]
        container = data.get("format", {})
        duration = container.get("duration", stream.get("duration"))
        if duration is None or not math.isfinite(float(duration)) or float(duration) < 0:
            raise MediaError("Media duration is unavailable")
        sample_rate = int(stream.get("sample_rate", 0))
        channels = int(stream.get("channels", 0))
        if sample_rate <= 0 or channels <= 0:
            raise MediaError("Audio sample rate or channel count is unavailable")
        return {
            "duration_ms": round(float(duration) * 1000),
            "sample_rate": sample_rate,
            "channels": channels,
            "channel_layout": stream.get("channel_layout", "unknown"),
            "format_name": container.get("format_name", path.suffix.lstrip(".")),
            "codec_name": stream.get("codec_name", "unknown"),
            "audio_stream_index": int(stream["index"]),
            "audio_streams": len(streams),
            "audio_start_ms": round(float(stream.get("start_time", 0)) * 1000),
            "container_start_ms": round(float(container.get("start_time", 0)) * 1000),
            "size_bytes": path.stat().st_size,
        }
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise MediaError("Invalid media metadata") from error


def extract_chunk(
    source: Path,
    target: Path,
    start_ms: int,
    end_ms: int,
    sample_rate: int = 16000,
    channel: int | None = None,
) -> None:
    """Extract a bounded PCM WAV without overwriting an existing target.

    ``channel=None`` explicitly requests a mixdown; an integer preserves one
    input microphone channel. Timestamps stay relative to original media and
    are supplied back to the ASR layer by the caller.
    """
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise MediaError("Source and target must be different")
    if not source.is_file():
        raise MediaError("Input is not a regular file")
    if target.exists():
        raise FileExistsError("Refusing to overwrite existing extraction")
    if start_ms < 0 or end_ms <= start_ms or sample_rate < 8000 or sample_rate > 192000:
        raise ValueError("Invalid extraction interval or sample rate")
    if channel is not None and (
        isinstance(channel, bool) or not isinstance(channel, int) or channel < 0
    ):
        raise ValueError("channel must be a nonnegative input channel index")
    if channel is not None and channel >= probe(source)["channels"]:
        raise ValueError("Selected channel does not exist in the first audio stream")
    target.parent.mkdir(parents=True, exist_ok=True)
    args = [
        _tool("ffmpeg"),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-protocol_whitelist",
        "file,pipe",
        "-ss",
        f"{start_ms / 1000:.3f}",
        "-i",
        str(source),
        "-t",
        f"{(end_ms - start_ms) / 1000:.3f}",
        "-map",
        "0:a:0",
        "-vn",
    ]
    # Preserve an audio stream's initial delay relative to the video timeline.
    # Without first_pts=0, FFmpeg may remove a leading no-audio interval and
    # make every subtitle early by the stream's offset.
    filters = ["aresample=async=1:first_pts=0"]
    if channel is not None:
        filters.append(f"pan=mono|c0=c{channel}")
    # A video can outlast its audio stream. Pad only the requested interval,
    # including chunks after the audio stream has ended, with actual silence.
    filters.append(f"apad=whole_dur={(end_ms - start_ms) / 1000:.3f}")
    args.extend(["-af", ",".join(filters)])
    args.extend(
        ["-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", "-f", "wav", str(target)]
    )
    _run(args)


def analyze_channels(path: Path, seconds: float = 10.0) -> dict:
    """Measure bounded sample energy and stereo correlation without identity claims.

    This is a diagnostic only: low correlation does not prove independent
    microphones; high correlation does not prove identical speaker content.
    """
    if not 0 < seconds <= 30:
        raise ValueError("Analysis sample must be between 0 and 30 seconds")
    info = probe(path)
    channels = info["channels"]
    if channels > 32:
        raise MediaError("Channel analysis supports at most 32 channels")
    raw = _run(
        [
            _tool("ffmpeg"),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(Path(path).resolve()),
            "-t",
            str(seconds),
            "-map",
            "0:a:0",
            "-ar",
            "8000",
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ],
        timeout=60,
    )
    samples = array.array("f")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    values = [samples[index::channels] for index in range(channels)]
    rms = [
        math.sqrt(sum(value * value for value in channel) / len(channel)) if channel else 0.0
        for channel in values
    ]
    correlation = None
    if channels == 2 and values[0] and values[1]:
        left, right = values
        count = min(len(left), len(right))
        left_mean, right_mean = sum(left) / count, sum(right) / count
        numerator = sum((left[i] - left_mean) * (right[i] - right_mean) for i in range(count))
        denominator = math.sqrt(
            sum((v - left_mean) ** 2 for v in left) * sum((v - right_mean) ** 2 for v in right)
        )
        if denominator > 0:
            correlation = max(-1.0, min(1.0, numerator / denominator))
    return {
        "channels": channels,
        "sample_duration_ms": min(info["duration_ms"], round(seconds * 1000)),
        "rms": [round(value, 6) for value in rms],
        "stereo_correlation": round(correlation, 6) if correlation is not None else None,
        "recommendation": "inspect_or_preserve_channels" if channels > 1 else "mono",
        "uncertainty": "Short-sample energy/correlation does not identify microphones or speakers.",
    }
