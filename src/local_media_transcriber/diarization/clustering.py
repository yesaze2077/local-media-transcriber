"""Global CAM++ clustering; anonymous acoustic clusters never establish identity."""

import math

AUTO_MAX_SPEAKERS = 50


def _graph_labels(matrix, neighbours=None):
    """Select a graph partition using a regularized relative eigengap.

    Positive cosine affinities retain similarity strength. A bounded kNN graph
    connects local variants without forcing every pair across a fixed distance
    cutoff. The normalized Laplacian distinguishes its large-scale components;
    the 0.05 denominator floor prevents tiny noisy eigenvalues deciding the count.
    This is an unsupervised heuristic, not a calibrated identity/count confidence.
    """
    import numpy as np
    from scipy.linalg import eigh
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, laplacian
    from scipy.sparse.linalg import ArpackNoConvergence, eigsh
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors

    count = len(matrix)
    if count == 1:
        return np.zeros(1, dtype=int), []
    neighbours = min(count - 1, neighbours or max(5, math.ceil(math.sqrt(count))))
    # Passing X=None excludes each sample itself, even when embeddings repeat.
    search = NearestNeighbors(n_neighbors=neighbours, metric="cosine", n_jobs=1).fit(matrix)
    distances, indices = search.kneighbors()
    affinity = np.maximum(0.0, 1.0 - distances) ** 2
    graph = csr_matrix(
        (affinity.ravel(), (np.repeat(np.arange(count), neighbours), indices.ravel())),
        shape=(count, count),
    )
    graph = graph.maximum(graph.T)
    graph.eliminate_zeros()
    components, component_labels = connected_components(graph)
    if components > AUTO_MAX_SPEAKERS:
        raise ValueError(
            "Automatic count exceeds 50 graph components; review audio or specify --speakers"
        )
    if components == count:
        return component_labels, [
            "Very little shared acoustic evidence; review the automatic count."
        ]
    normalized = laplacian(graph, normed=True)
    limit = min(AUTO_MAX_SPEAKERS, count - 1)
    if count <= 128:
        values, vectors = eigh(normalized.toarray(), subset_by_index=[0, limit])
    else:
        try:
            values, vectors = eigsh(
                normalized,
                k=limit + 1,
                which="SM",
                tol=1e-5,
                v0=np.linspace(1.0, 2.0, count),
                maxiter=max(1000, 5 * count),
            )
        except ArpackNoConvergence as error:
            raise RuntimeError("Automatic count did not converge; specify --speakers") from error
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
    values = np.maximum(values, 0.0)
    scores = np.diff(values) / np.maximum(values[1:], 0.05)
    selected = int(np.argmax(scores)) + 1
    # A disconnected graph cannot support fewer labels than its components.
    selected = max(selected, components)
    warnings = []
    if selected == 1:
        return np.zeros(count, dtype=int), warnings
    if selected == components:
        labels = component_labels
    else:
        embedding = vectors[:, :selected]
        embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
        labels = KMeans(n_clusters=selected, n_init=10, random_state=0).fit_predict(embedding)
    if np.partition(scores, -2)[-1] - np.partition(scores, -2)[-2] < 0.05:
        warnings.append(
            "Automatic count has competing graph partitions; specify a known count and review."
        )
    if np.bincount(labels).min() < 3:
        warnings.append(
            "Some speaker clusters have fewer than three windows; short voices/counts need review."
        )
    return labels, warnings


def _coherent_duration(segments):
    """Longest overlapping/touching run on one channel, not summed overlap time."""
    longest = 0
    previous_channel = object()
    beginning = ending = 0
    for segment in sorted(segments, key=lambda item: (str(item.get("channel")), item["start_ms"])):
        start, end = segment["start_ms"], segment["end_ms"]
        channel = segment.get("channel")
        if channel != previous_channel or start > ending:
            beginning, ending = start, end
        else:
            ending = max(ending, end)
        longest = max(longest, ending - beginning)
        previous_channel = channel
    return longest


def _automatic_labels(matrix, segments=None):
    """Estimate the broad partition, then check coherent minority voices.

    A sqrt(N) neighbourhood can over-smooth a few seconds of a third voice.
    A three-neighbour view proposes extra groups, but it cannot directly decide
    the count: phonetic modes and transition windows also fragment that view.
    Fine-group promotion requires >=3 windows, >=3 seconds of coherent support,
    a compact centroid, and novelty relative to every existing voice centroid.
    A subsequent tiny-pair check additionally requires local audio/alignment
    evidence. These are reviewable heuristics, not a short-turn guarantee.
    """
    import numpy as np

    labels, warnings = _graph_labels(matrix)
    if segments is None or len(matrix) < 6:
        return labels, warnings
    try:
        fine, _ = _graph_labels(matrix, neighbours=3)
    except (ValueError, RuntimeError):
        return labels, warnings + [
            "Fine-scale count check unavailable; review short contributions."
        ]
    promoted = 0
    groups = np.unique(fine) if len(np.unique(fine)) > len(np.unique(labels)) else []
    for group in sorted(groups, key=lambda value: -np.count_nonzero(fine == value)):
        ids = np.flatnonzero(fine == group)
        if len(ids) < 3:
            continue
        parent, counts = np.unique(labels[ids], return_counts=True)
        parent = parent[int(np.argmax(counts))]
        if counts.max() / len(ids) < 0.8:
            continue
        # Only split the dominant parent; preserve other already-established
        # groups that the finer partition happened to mix into this candidate.
        ids = ids[labels[ids] == parent]
        if (
            len(ids) < 3
            or np.count_nonzero(labels == parent) - len(ids) < 3
            or _coherent_duration([segments[i] for i in ids]) < 3000
        ):
            continue
        center = matrix[ids].mean(axis=0)
        center /= max(np.linalg.norm(center), 1e-12)
        radius = float(np.mean(1.0 - matrix[ids] @ center))
        centers = np.stack([matrix[labels == value].mean(axis=0) for value in np.unique(labels)])
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        if radius > 0.2 or np.min(1.0 - centers @ center) < 0.2:
            continue
        if len(centers) >= AUTO_MAX_SPEAKERS:
            warnings.append(
                "Automatic count reached 50 candidates; review remaining short contributions."
            )
            break
        labels[ids] = int(labels.max()) + 1
        promoted += 1
    if promoted:
        warnings.append(
            f"Fine-scale check retained {promoted} coherent minority acoustic groups; verify the automatic count."
        )
    # Run this only after the existing longer minority groups are established.
    from .short_voice import promote_short_pairs

    labels, details = promote_short_pairs(matrix, labels, segments, AUTO_MAX_SPEAKERS)
    warnings.extend(details)
    return labels, warnings


def assign_speakers(
    segments: list[dict], embeddings: list, speakers: int | None = None
) -> tuple[list[dict], list[str]]:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    if len(segments) != len(embeddings):
        raise ValueError("Segments and embeddings must have matching lengths")
    if speakers is not None and (
        isinstance(speakers, bool) or not isinstance(speakers, int) or speakers < 1
    ):
        raise ValueError("speakers must be a positive integer")
    warnings = [
        "CAM++ labels are acoustic clusters, not real identities; overlap is not separated."
    ]
    eligible = []
    invalid = 0
    dimension = None
    for index, (segment, value) in enumerate(zip(segments, embeddings)):
        segment["speaker_id"] = None
        if value is None:
            continue
        vector = np.asarray(value, dtype="float64")
        if vector.ndim != 1 or not vector.size:
            raise ValueError("Speaker embeddings must be nonempty one-dimensional vectors")
        if not np.isfinite(vector).all() or not np.any(vector):
            invalid += 1
            continue
        if dimension is not None and len(vector) != dimension:
            raise ValueError("Speaker embedding dimensions differ")
        dimension = len(vector)
        # Scaling before the norm avoids overflow and keeps magnitude irrelevant.
        vector = vector / np.max(np.abs(vector))
        vector /= np.linalg.norm(vector)
        eligible.append((index, vector))
    if invalid:
        warnings.append(
            f"Ignored {invalid} nonfinite or zero speaker embeddings; their labels remain unknown."
        )
    if not eligible:
        return [], warnings + ["No reliable speaker embeddings were available."]
    if len(eligible) > 5000:
        raise ValueError(
            "Global clustering supports up to 5000 speaker windows per job; split the job or disable diarization"
        )
    matrix = np.stack([value for _, value in eligible])
    if speakers is None and len(matrix) < 3:
        warnings.append(
            "Fewer than three usable windows; automatic count has insufficient evidence. Specify --speakers when known."
        )
    if speakers and speakers > len(matrix):
        warnings.append(
            "Requested speaker count exceeds usable segments; count was bounded by evidence."
        )
    if len(matrix) == 1 or speakers == 1:
        labels = np.zeros(len(matrix), dtype=int)
    elif speakers is not None:
        # Preserve the explicitly constrained v0.1.0 partitioning path.
        labels = AgglomerativeClustering(
            n_clusters=min(speakers, len(matrix)), metric="cosine", linkage="average"
        ).fit_predict(matrix)
    else:
        labels, details = _automatic_labels(matrix, [segments[index] for index, _ in eligible])
        warnings.extend(details)
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
    if speakers is None:
        warnings.append(
            "Automatic count uses multiscale cosine-neighbour graphs and relative eigengaps; review labels manually."
        )
    return stats, warnings
