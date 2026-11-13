"""Bounded local window evidence; energy/alignment are not speaker identity truth."""

import math
from numbers import Real


def interval_support_ms(intervals, start_ms, end_ms):
    """Union coverage inside a measured interval; never sum overlapping tokens twice."""
    if intervals is None or end_ms <= start_ms:
        return 0
    clipped = []
    for pair in intervals:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        a, b = pair[:2]
        if not isinstance(a, Real) or not isinstance(b, Real):
            continue
        if not math.isfinite(a) or not math.isfinite(b):
            continue
        a, b = max(start_ms, int(round(a))), min(end_ms, int(round(b)))
        if b > a:
            clipped.append((a, b))
    total, previous = 0, None
    for a, b in sorted(clipped):
        if previous is None or a > previous[1]:
            if previous is not None:
                total += previous[1] - previous[0]
            previous = (a, b)
        else:
            previous = (previous[0], max(previous[1], b))
    return total + (previous[1] - previous[0] if previous is not None else 0)


def window_evidence(audio, sample_rate, start_ms, aligned_intervals_ms=None):
    """Measure active 20ms energy frames and retain available ASR interval support.

    The relative energy floor adapts to each window's loudness. This measures
    nonsilence, not speech presence. The separate alignment evidence is required
    by short-voice promotion; absent timing remains None instead of invented data.
    """
    import numpy as np

    values = np.asarray(audio, dtype="float64")
    if values.ndim != 1 or not len(values) or sample_rate <= 0 or not np.isfinite(values).all():
        raise ValueError("Window evidence requires finite mono audio and a valid sample rate")
    frame_size = max(1, round(sample_rate * 0.02))
    threshold = max(1e-4, float(np.quantile(np.abs(values), 0.95)) * 0.03)
    active = 0
    for begin in range(0, len(values), frame_size):
        frame = values[begin : begin + frame_size]
        if np.sqrt(np.mean(frame * frame)) > threshold:
            active += len(frame)
    end_ms = start_ms + round(len(values) / sample_rate * 1000)
    clipped = None
    if aligned_intervals_ms is not None:
        clipped = []
        for pair in aligned_intervals_ms:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            a, b = pair[:2]
            if (
                not isinstance(a, Real)
                or not isinstance(b, Real)
                or not math.isfinite(a)
                or not math.isfinite(b)
            ):
                continue
            a, b = max(start_ms, int(round(a))), min(end_ms, int(round(b)))
            if b > a:
                clipped.append([a, b])
    return {
        "quality_version": 1,
        "activity_fraction": active / len(values),
        "aligned_intervals_ms": clipped,
    }
