"""Numerical clustering contracts; no media, model downloads or voiceprint fixtures."""

from copy import deepcopy

import numpy as np
import pytest

from local_media_transcriber.diarization.clustering import assign_speakers


def _segments(count):
    return [
        {"start_ms": i * 1000, "end_ms": (i + 1) * 1000, "speaker_id": None} for i in range(count)
    ]


def _separated_groups(count):
    # Deliberately simple artificial geometry, not generated speaker embeddings.
    rng = np.random.default_rng(8201)
    return np.concatenate(
        [np.eye(5)[index] + rng.normal(0, 0.008, (6, 5)) for index in range(count)]
    )


def _partition(segments):
    labels = np.asarray([segment["speaker_id"] for segment in segments])
    return labels[:, None] == labels[None, :]


@pytest.mark.parametrize("expected", [1, 2, 3])
def test_automatic_count_distinguishes_simple_separated_groups(expected):
    vectors = _separated_groups(expected)
    segments = _segments(len(vectors))
    stats, _ = assign_speakers(segments, vectors.tolist())
    assert len(stats) == expected
    assert sum(stat["segment_count"] for stat in stats) == len(vectors)
    for begin in range(0, len(vectors), 6):
        assert len({s["speaker_id"] for s in segments[begin : begin + 6]}) == 1


def test_positive_vector_scale_does_not_change_partition():
    vectors = _separated_groups(3)
    original = _segments(len(vectors))
    scaled = deepcopy(original)
    assign_speakers(original, vectors.tolist())
    scales = np.linspace(0.05, 20, len(vectors))[:, None]
    assign_speakers(scaled, (vectors * scales).tolist())
    np.testing.assert_array_equal(_partition(original), _partition(scaled))


def test_window_order_does_not_change_membership_up_to_label_permutation():
    vectors = _separated_groups(3)
    original = _segments(len(vectors))
    assign_speakers(original, vectors.tolist())
    order = np.random.default_rng(934).permutation(len(vectors))
    shuffled = [deepcopy(_segments(len(vectors))[index]) for index in order]
    assign_speakers(shuffled, vectors[order].tolist())
    restored = [shuffled[index] for index in np.argsort(order)]
    np.testing.assert_array_equal(_partition(original), _partition(restored))


def test_missing_embeddings_are_not_counted_as_speakers():
    segments = _segments(3)
    stats, warnings = assign_speakers(segments, [None, [1.0, 0.0], None])
    assert len(stats) == 1
    assert segments[0]["speaker_id"] is None and segments[2]["speaker_id"] is None
    assert segments[1]["speaker_id"] == stats[0]["id"]
    assert isinstance(warnings, list)


def test_no_embeddings_does_not_invent_a_speaker():
    segments = _segments(3)
    stats, warnings = assign_speakers(segments, [None, None, None])
    assert stats == [] and warnings
    assert all(segment["speaker_id"] is None for segment in segments)


@pytest.mark.parametrize("requested", [1, 2, 3])
def test_explicit_count_remains_a_separate_user_constraint(requested):
    vectors = _separated_groups(3)
    segments = _segments(len(vectors))
    stats, _ = assign_speakers(segments, vectors.tolist(), speakers=requested)
    assert len(stats) == requested
    assert all(segment["speaker_id"] in {s["id"] for s in stats} for segment in segments)


@pytest.mark.parametrize("invalid", [[0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0]])
def test_invalid_embedding_never_becomes_a_reported_speaker(invalid):
    segments = _segments(1)
    # Rejecting malformed input or excluding invalid evidence are both valid contracts.
    try:
        stats, _ = assign_speakers(segments, [invalid])
    except ValueError:
        return
    assert stats == []
    assert segments[0]["speaker_id"] is None


@pytest.mark.parametrize("expected", [2, 3])
def test_multiscale_groups_are_not_counted_as_separate_small_subgroups(expected):
    """Artificial geometry: two smaller modes per well separated macro group.

    These vectors are made from orthogonal axes and are not voiceprint data or
    a simulation of measured speaker embeddings. They guard against fixing the
    observed oversegmentation by globally forcing every input to two clusters.
    """
    rng = np.random.default_rng(318)
    axes = np.eye(expected * 2)
    angle = np.deg2rad(30)
    groups = []
    for index in range(expected):
        for sign in (-1, 1):
            center = np.cos(angle) * axes[index] + sign * np.sin(angle) * axes[expected + index]
            groups.append(center + rng.normal(0, 0.002, (5, len(center))))
    vectors = np.concatenate(groups)
    segments = _segments(len(vectors))
    stats, _ = assign_speakers(segments, vectors.tolist())
    assert len(stats) == expected
    for begin in range(0, len(vectors), 10):
        assert len({s["speaker_id"] for s in segments[begin : begin + 10]}) == 1


def test_sparse_graph_path_preserves_three_groups():
    rng = np.random.default_rng(31)
    vectors = np.concatenate([np.eye(6)[i] + rng.normal(0, 0.015, (50, 6)) for i in range(3)])
    segments = _segments(len(vectors))
    stats, _ = assign_speakers(segments, vectors.tolist())
    assert len(stats) == 3
    for start in range(0, len(vectors), 50):
        assert len({s["speaker_id"] for s in segments[start : start + 50]}) == 1


def test_invalid_reused_window_does_not_keep_its_old_label():
    segments = [{"start_ms": 0, "end_ms": 1000, "speaker_id": "SPEAKER_99"}]
    stats, _ = assign_speakers(segments, [None])
    assert stats == [] and segments[0]["speaker_id"] is None


def test_window_limit_is_enforced_before_building_a_graph():
    with pytest.raises(ValueError, match="5000"):
        assign_speakers(_segments(5001), [[1.0, 0.0]] * 5001)


def test_short_coherent_third_group_is_not_smoothed_into_a_long_group():
    # Artificial unbalanced groups: a short third mode resembles one long mode,
    # but remains distinct and lasts several overlapping windows.
    rng = np.random.default_rng(920)
    a, b = np.eye(4)[:2]
    third = 0.7 * a + np.sqrt(1 - 0.7**2) * np.eye(4)[2]
    vectors = np.concatenate(
        [
            a + rng.normal(0, 0.004, (30, 4)),
            b + rng.normal(0, 0.004, (30, 4)),
            third + rng.normal(0, 0.004, (6, 4)),
        ]
    )
    segments = [
        {"start_ms": i * 750, "end_ms": i * 750 + 1500, "speaker_id": None}
        for i in range(len(vectors))
    ]
    stats, _ = assign_speakers(segments, vectors.tolist())
    assert len(stats) == 3
    assert len({s["speaker_id"] for s in segments[-6:]}) == 1
    assert segments[-1]["speaker_id"] != segments[0]["speaker_id"]


@pytest.mark.parametrize(
    "windows,expected",
    [
        (
            [
                {"start_ms": 0, "end_ms": 1500},
                {"start_ms": 750, "end_ms": 2250},
                {"start_ms": 1500, "end_ms": 3000},
            ],
            3000,
        ),
        ([{"start_ms": 0, "end_ms": 1500}, {"start_ms": 5000, "end_ms": 6500}], 1500),
        (
            [
                {"start_ms": 0, "end_ms": 1500, "channel": 0},
                {"start_ms": 750, "end_ms": 2250, "channel": 1},
            ],
            1500,
        ),
    ],
)
def test_minority_evidence_uses_coherent_time_not_summed_overlaps(windows, expected):
    from local_media_transcriber.diarization.clustering import _coherent_duration

    assert _coherent_duration(windows) == expected


def test_touching_speech_windows_are_continuous_support():
    from local_media_transcriber.diarization.clustering import _coherent_duration

    windows = [{"start_ms": i * 1500, "end_ms": (i + 1) * 1500} for i in range(3)]
    assert _coherent_duration(windows) == 4500


def test_minority_promotion_does_not_consume_an_existing_small_speaker(monkeypatch):
    from local_media_transcriber.diarization import clustering

    base = np.array([0] * 9 + [1] + [2] * 9)
    fine = np.array([0] * 5 + [1] * 4 + [0] + [2] * 5 + [3] * 4)
    axes = np.eye(4)
    matrix = np.array([axes[1]] * 5 + [axes[0]] * 4 + [axes[2]] + [axes[3]] * 9)
    answers = iter([(base.copy(), []), (fine.copy(), [])])
    monkeypatch.setattr(clustering, "_graph_labels", lambda *a, **kw: next(answers))
    labels, _ = clustering._automatic_labels(
        matrix, [{"start_ms": i * 750, "end_ms": i * 750 + 1500} for i in range(19)]
    )
    assert labels[9] == 1
    assert len(np.unique(labels)) == 4
