"""Conservative tiny-voice evidence, applied after broad and fine graph partitions."""

import math
from ..audio.quality import interval_support_ms


def _usable(window):
    fraction = window.get("activity_fraction")
    return (
        window.get("quality_version") == 1
        and isinstance(fraction, (int, float))
        and math.isfinite(fraction)
        and fraction >= 0.6
        and window["end_ms"] - window["start_ms"] >= 1000
        and isinstance(window.get("aligned_intervals_ms"), list)
    )


def promote_short_pairs(matrix, labels, windows, max_speakers=50):
    """Propose a novel pair only with audio energy and shared aligned tokens.

    Two overlapping observations are correlated; they do not become two seconds
    of independent speech. Require a measured time union, different starts and
    common ASR support instead. No energy/alignment metadata means no promotion.
    """
    import numpy as np

    usable = [i for i, w in enumerate(windows) if _usable(w)]
    usable.sort(key=lambda i: (str(windows[i].get("channel")), windows[i]["start_ms"]))
    promoted = 0
    for i, j in zip(usable, usable[1:]):
        a, b = windows[i], windows[j]
        offset = b["start_ms"] - a["start_ms"]
        left, right = max(a["start_ms"], b["start_ms"]), min(a["end_ms"], b["end_ms"])
        if (
            a.get("channel") != b.get("channel")
            or not 250 <= offset <= 1000
            or not 300 <= right - left <= 1000
        ):
            continue
        if labels[i] != labels[j] or np.count_nonzero(labels == labels[i]) - 2 < 3:
            continue
        if max(a["end_ms"], b["end_ms"]) - min(a["start_ms"], b["start_ms"]) < 1500:
            continue
        if (
            min(
                interval_support_ms(a["aligned_intervals_ms"], left, right),
                interval_support_ms(b["aligned_intervals_ms"], left, right),
            )
            < 200
        ):
            continue
        if 1.0 - matrix[i] @ matrix[j] > 0.25:
            continue
        center = matrix[[i, j]].mean(axis=0)
        center /= max(np.linalg.norm(center), 1e-12)
        centers = np.stack([matrix[labels == v].mean(axis=0) for v in np.unique(labels)])
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        if np.min(1.0 - centers @ center) < 0.25:
            continue
        if len(centers) >= max_speakers:
            return labels, [
                "Short-voice check reached the candidate limit; review unresolved contributions."
            ]
        labels[[i, j]] = int(labels.max()) + 1
        promoted += 1
    warnings = []
    if promoted:
        warnings.append(
            f"Audio/alignment evidence retained {promoted} tiny acoustic groups; verify count and boundaries manually."
        )
    return labels, warnings
