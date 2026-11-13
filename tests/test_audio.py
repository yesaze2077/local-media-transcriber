"""Real FFmpeg preprocessing tests with locally generated tones, not ASR claims."""

import hashlib
import math
import shutil
import struct
import subprocess
import wave

import pytest

from local_media_transcriber.audio import MediaError, analyze_channels, extract_chunk, probe


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)


def make_stereo(path, seconds=1):
    rate = 16000
    samples = bytearray()
    for index in range(round(rate * seconds)):
        left = round(10000 * math.sin(2 * math.pi * 440 * index / rate))
        right = round(6000 * math.sin(2 * math.pi * 880 * index / rate))
        samples.extend(struct.pack("<hh", left, right))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples)


def test_probe_and_extract_preserve_source_and_channel(tmp_path):
    source = tmp_path / "stereo.wav"
    make_stereo(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = probe(source)
    assert metadata["duration_ms"] == 1000
    assert metadata["channels"] == 2
    assert metadata["sample_rate"] == 16000
    target = tmp_path / "chunk.wav"
    extract_chunk(source, target, 250, 750, 22050, channel=1)
    result = probe(target)
    assert result["channels"] == 1
    assert result["sample_rate"] == 22050
    assert abs(result["duration_ms"] - 500) <= 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(FileExistsError):
        extract_chunk(source, target, 0, 500)


def test_channel_diagnostic_reports_independent_tone_sample(tmp_path):
    source = tmp_path / "stereo.wav"
    make_stereo(source)
    result = analyze_channels(source)
    assert result["channels"] == 2
    assert len(result["rms"]) == 2
    assert result["rms"][0] > result["rms"][1] > 0
    assert abs(result["stereo_correlation"]) < 0.05
    assert result["recommendation"] == "inspect_or_preserve_channels"


def test_invalid_file_is_not_media(tmp_path):
    source = tmp_path / "invalid.mp3"
    source.write_bytes(b"this is not a media container")
    with pytest.raises(MediaError):
        probe(source)


@pytest.mark.parametrize("channel", [-1, True, "0", 2])
def test_reject_invalid_channel(tmp_path, channel):
    source = tmp_path / "stereo.wav"
    make_stereo(source)
    with pytest.raises(ValueError):
        extract_chunk(source, tmp_path / "output.wav", 0, 500, channel=channel)


def test_silence_analysis(tmp_path):
    source = tmp_path / "silence.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 16000)
    result = analyze_channels(source)
    assert result["rms"] == [0.0]
    assert result["stereo_correlation"] is None


def test_video_audio_offset_preserves_original_timeline(tmp_path):
    source = tmp_path / "delayed_audio.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:r=10:d=5",
            "-itsoffset",
            "2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=3",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    assert probe(source)["audio_start_ms"] > 1800
    target = tmp_path / "extracted.wav"
    extract_chunk(source, target, 0, 5000)
    assert probe(target)["duration_ms"] == 5000
    with wave.open(str(target), "rb") as audio:
        assert audio.readframes(16000) == b"\x00\x00" * 16000


@pytest.mark.parametrize(
    "suffix,codec,video_codec",
    [
        ("wav", "pcm_s16le", None),
        ("mp3", "libmp3lame", None),
        ("m4a", "aac", None),
        ("aac", "aac", None),
        ("flac", "flac", None),
        ("ogg", "libopus", None),
        ("opus", "libopus", None),
        ("wma", "wmav2", None),
        ("mp4", "aac", "mpeg4"),
        ("mov", "aac", "mpeg4"),
        ("mkv", "flac", "mpeg4"),
        ("webm", "libopus", "libvpx"),
        ("avi", "pcm_s16le", "mpeg4"),
        ("m4v", "aac", "mpeg4"),
    ],
)
def test_real_container_codec_preprocessing_matrix(tmp_path, suffix, codec, video_codec):
    """Verify media decoding, not language recognition, for each input format."""
    source = tmp_path / "source.wav"
    make_stereo(source)
    encoded = tmp_path / f"input.{suffix}"
    args = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(source)]
    if video_codec:
        args += [
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=5:d=1",
            "-c:v",
            video_codec,
            "-shortest",
        ]
    args += ["-c:a", codec]
    # The .m4v extension otherwise selects FFmpeg's raw MPEG-4 video muxer.
    if suffix == "m4v":
        args += ["-f", "mp4"]
    args += [str(encoded)]
    subprocess.run(args, check=True)
    assert probe(encoded)["duration_ms"] > 0
    target = tmp_path / "decoded.wav"
    extract_chunk(encoded, target, 0, 500)
    info = probe(target)
    assert info["channels"] == 1 and info["sample_rate"] == 16000
    assert abs(info["duration_ms"] - 500) <= 50


def test_video_silent_tail_extracts_as_silence(tmp_path):
    source = tmp_path / "short_audio.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:r=10:d=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=2",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    target = tmp_path / "tail.wav"
    extract_chunk(source, target, 3000, 4000)
    assert probe(target)["duration_ms"] == 1000
    with wave.open(str(target), "rb") as audio:
        assert audio.readframes(16000) == b"\x00\x00" * 16000
