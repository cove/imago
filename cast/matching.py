from __future__ import annotations

from typing import Any

import numpy as np


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
    buckets: dict[str, list[np.ndarray]] = {}
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
        buckets.setdefault(person_id, []).append(vector)

    out: dict[str, dict[str, Any]] = {}
    for person_id, vectors in buckets.items():
        if not vectors:
            continue
        dims = {tuple(vec.shape) for vec in vectors}
        if len(dims) != 1:
            continue
        stacked = np.vstack(vectors)
        proto = normalize_embedding(stacked.mean(axis=0))
        out[person_id] = {
            "embedding": proto,
            "count": int(stacked.shape[0]),
            "dimension": int(stacked.shape[1]),
        }
    return out


def suggest_people(
    *,
    query_embedding: Any,
    faces: list[dict[str, Any]],
    top_k: int = 3,
    min_similarity: float = -1.0,
) -> list[dict[str, Any]]:
    query = normalize_embedding(parse_embedding(query_embedding))
    prototypes = build_person_prototypes(faces)
    results: list[dict[str, Any]] = []
    for person_id, details in prototypes.items():
        embedding = details.get("embedding")
        if not isinstance(embedding, np.ndarray):
            continue
        if embedding.shape != query.shape:
            continue
        score = float(np.dot(query, embedding))
        if score < float(min_similarity):
            continue
        results.append(
            {
                "person_id": str(person_id),
                "score": score,
                "sample_count": int(details.get("count", 0)),
            }
        )
    results.sort(key=lambda row: row["score"], reverse=True)
    return results[: max(1, int(top_k))]
