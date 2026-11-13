"""Global CAM++ clustering across bounded audio chunks; labels never imply identity."""


def assign_speakers(
    segments: list[dict], embeddings: list, speakers: int | None = None
) -> tuple[list[dict], list[str]]:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    warnings = [
        "CAM++ labels are acoustic clusters, not real identities; overlap is not separated."
    ]
    eligible = [
        (i, np.asarray(value, dtype="float32"))
        for i, value in enumerate(embeddings)
        if value is not None
    ]
    if not eligible:
        return [], warnings + ["No reliable speaker embeddings were available."]
    if len(eligible) > 5000:
        raise ValueError(
            "Global clustering supports up to 5000 speaker windows per job; split the job or disable diarization"
        )
    matrix = np.stack([value for _, value in eligible])
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-10)
    if speakers and speakers > len(matrix):
        warnings.append(
            "Requested speaker count exceeds usable segments; count was bounded by evidence."
        )
    if len(matrix) == 1 or speakers == 1:
        labels = np.zeros(len(matrix), dtype=int)
    else:
        # Custom bounded global clustering rather than independent per-chunk labels.
        # The threshold is heuristic and is not a diarization accuracy guarantee.
        model = AgglomerativeClustering(
            n_clusters=min(speakers, len(matrix)) if speakers else None,
            metric="cosine",
            linkage="average",
            distance_threshold=None if speakers else 0.35,
        )
        labels = model.fit_predict(matrix)
    stable = {}
    for (index, _), label in zip(eligible, labels):
        stable.setdefault(int(label), f"SPEAKER_{len(stable):02d}")
        segments[index]["speaker_id"] = stable[int(label)]
    stats = []
    for identifier in stable.values():
        matched = [s for s in segments if s["speaker_id"] == identifier]
        stats.append(
            {
                "id": identifier,
                "display_name": identifier,
                "segment_count": len(matched),
                "duration_ms": sum(s["end_ms"] - s["start_ms"] for s in matched),
            }
        )
    if not speakers:
        warnings.append(
            "Automatic speaker count uses a heuristic cosine threshold; review labels manually."
        )
    return stats, warnings
