"""Artificial numeric/audio evidence contracts, not voiceprints or ASR accuracy tests."""

import numpy as np
import pytest
from local_media_transcriber.diarization.clustering import assign_speakers
from local_media_transcriber.audio.quality import window_evidence, interval_support_ms


def tiny_case():
    rng = np.random.default_rng(42)
    axes = np.eye(4)
    center = 0.6 * axes[0] + 0.8 * axes[2]
    vectors = np.concatenate(
        [
            axes[0] + rng.normal(0, 0.003, (30, 4)),
            axes[1] + rng.normal(0, 0.003, (30, 4)),
            center + rng.normal(0, 0.003, (2, 4)),
        ]
    )
    windows = [
        {"start_ms": i * 750, "end_ms": i * 750 + 1500, "speaker_id": None}
        for i in range(len(vectors))
    ]
    for w in windows[-2:]:
        w.update(quality_version=1, activity_fraction=0.8, aligned_intervals_ms=[[45000, 46500]])
    return windows, vectors.tolist()


def test_quality_supported_tiny_third_voice_is_not_absorbed():
    windows, vectors = tiny_case()
    stats, _ = assign_speakers(windows, vectors)
    assert len(stats) == 3
    assert windows[-1]["speaker_id"] == windows[-2]["speaker_id"]
    assert windows[-1]["speaker_id"] != windows[0]["speaker_id"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_quality",
        "silence",
        "no_alignment",
        "no_shared_tokens",
        "duplicate_time",
        "channel_change",
    ],
)
def test_short_voice_requires_independent_quality_and_temporal_evidence(change):
    windows, vectors = tiny_case()
    for w in windows[-2:]:
        if change == "missing_quality":
            w.pop("quality_version")
        if change == "silence":
            w["activity_fraction"] = 0.1
        if change == "no_alignment":
            w["aligned_intervals_ms"] = None
        if change == "no_shared_tokens":
            w["aligned_intervals_ms"] = [[44000, 44500]]
    if change == "duplicate_time":
        windows[-1].update(start_ms=windows[-2]["start_ms"], end_ms=windows[-2]["end_ms"])
    if change == "channel_change":
        windows[-1]["channel"] = 1
    stats, _ = assign_speakers(windows, vectors)
    assert len(stats) == 2


def test_window_energy_and_missing_alignment_do_not_invent_speech():
    silent = window_evidence(np.zeros(24000), 16000, 1000, None)
    assert silent["activity_fraction"] == 0 and silent["aligned_intervals_ms"] is None
    signal = np.sin(np.arange(24000) * 0.1) * 0.1
    active = window_evidence(signal, 16000, 1000, [[900, 1100], [1200, 1400]])
    assert active["activity_fraction"] > 0.9
    assert active["aligned_intervals_ms"] == [[1000, 1100], [1200, 1400]]


def test_token_coverage_uses_union_and_clips_to_common_interval():
    assert interval_support_ms([[100, 400], [200, 600], [800, 900]], 250, 850) == 400
    assert interval_support_ms(None, 0, 1000) == 0


def test_scaled_audio_keeps_relative_activity_measurement():
    wave = np.concatenate([np.zeros(6400), np.sin(np.arange(17600) * 0.1) * 0.1])
    a = window_evidence(wave, 16000, 0)
    b = window_evidence(wave * 0.5, 16000, 0)
    assert a["activity_fraction"] == b["activity_fraction"]
