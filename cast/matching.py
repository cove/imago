from __future__ import annotations

from typing import Any

import numpy as np

_MAX_SAMPLES_PER_PERSON = 64
_EXEMPLAR_COUNT = 40
_PROTOTYPE_WEIGHT = 0.20
_EXEMPLAR_WEIGHT = 0.80
_MAX_COUNT_BOOST = 0.005
_COUNT_BOOST_CAP = 12


def parse_embedding(raw: Any) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        arr = raw.astype(np.float32).reshape(-1)
        return arr
    if isinstance(raw, (list, tuple)):
        values = [float(item) for item in raw]
        return np.asarray(values, dtype=np.float32).reshape(-1)
    if isinstance(raw, str):
        text = raw.replace("\n", ",").replace("\t", ",")
        parts = [piece.strip() for piece in text.split(",")]
        values = [float(piece) for piece in parts if piece]
        return np.asarray(values, dtype=np.float32).reshape(-1)
    raise ValueError("Embedding must be an array-like value or comma-separated string.")


def normalize_embedding(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Embedding cannot be empty.")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("Embedding norm is zero.")
    return arr / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    left = normalize_embedding(a)
    right = normalize_embedding(b)
    if left.shape != right.shape:
        raise ValueError("Embeddings must have the same dimensions.")
    return float(np.dot(left, right))


def build_person_prototypes(
    faces: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[tuple[float, np.ndarray]]] = {}
    for face in list(faces or []):
        if not isinstance(face, dict):
            continue
        person_id = str(face.get("person_id") or "").strip()
        if not person_id:
            continue
        emb = face.get("embedding")
        try:
            vector = normalize_embedding(parse_embedding(emb))
        except Exception:
            continue
        raw_quality = face.get("quality")
        try:
            quality = float(raw_quality) if raw_quality is not None else 0.5
        except Exception:
            quality = 0.5
        quality = max(0.0, min(1.0, quality))
        buckets.setdefault(person_id, []).append((quality, vector))

    out: dict[str, dict[str, Any]] = {}
    for person_id, pairs in buckets.items():
        if not pairs:
            continue
        ranked = sorted(pairs, key=lambda row: row[0], reverse=True)[:_MAX_SAMPLES_PER_PERSON]
        vectors = [row[1] for row in ranked]
        dims = {tuple(vec.shape) for vec in vectors}
        if len(dims) != 1:
            continue
        stacked = np.vstack(vectors)
        weights = np.asarray(
            [0.35 + (0.65 * float(row[0])) for row in ranked],
            dtype=np.float32,
        ).reshape(-1, 1)
        weighted_mean = np.sum(stacked * weights, axis=0)
        proto = normalize_embedding(weighted_mean)
        exemplars = [row[1] for row in ranked[:_EXEMPLAR_COUNT]]
        out[person_id] = {
            "embedding": proto,
            "count": int(len(ranked)),
            "dimension": int(stacked.shape[1]),
            "exemplars": exemplars,
        }
    return out


def suggest_people(
    *,
    query_embedding: Any,
    faces: list[dict[str, Any]],
    top_k: int = 3,
    min_similarity: float = -1.0,
) -> list[dict[str, Any]]:
    prototypes = build_person_prototypes(faces)
    return suggest_people_from_prototypes(
        query_embedding=query_embedding,
        prototypes=prototypes,
        top_k=top_k,
        min_similarity=min_similarity,
    )


def suggest_people_from_prototypes(
    *,
    query_embedding: Any,
    prototypes: dict[str, dict[str, Any]],
    top_k: int = 3,
    min_similarity: float = -1.0,
) -> list[dict[str, Any]]:
    query = normalize_embedding(parse_embedding(query_embedding))
    results: list[dict[str, Any]] = []
    for person_id, details in prototypes.items():
        embedding = details.get("embedding")
        if not isinstance(embedding, np.ndarray):
            continue
        if embedding.shape != query.shape:
            continue
        prototype_score = float(np.dot(query, embedding))
        exemplar_vectors = list(details.get("exemplars") or [])
        exemplar_scores: list[float] = []
        for candidate in exemplar_vectors:
            if not isinstance(candidate, np.ndarray):
                continue
            if candidate.shape != query.shape:
                continue
            exemplar_scores.append(float(np.dot(query, candidate)))
        if exemplar_scores:
            exemplar_score = float(max(exemplar_scores))
        else:
            exemplar_score = prototype_score

        sample_count = int(details.get("count", 0))
        normalized_count = float(min(_COUNT_BOOST_CAP, max(0, sample_count))) / float(_COUNT_BOOST_CAP)
        count_boost = normalized_count * float(_MAX_COUNT_BOOST)
        score = (
            (float(_PROTOTYPE_WEIGHT) * prototype_score)
            + (float(_EXEMPLAR_WEIGHT) * exemplar_score)
            + count_boost
        )
        if score < float(min_similarity):
            continue
        results.append(
            {
                "person_id": str(person_id),
                "score": score,
                "sample_count": sample_count,
            }
        )
    results.sort(key=lambda row: row["score"], reverse=True)
    return results[: max(1, int(top_k))]
