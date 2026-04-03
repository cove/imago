from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .matching import (
    choose_suggested_candidate,
    face_embedding_model,
    normalize_embedding,
    parse_embedding,
    suggest_people_from_prototypes,
)

_SIMILARITY_BATCH_SIZE = 256


def _coerce_quality(value: Any) -> float:
    try:
        quality = float(value) if value is not None else 0.0
    except Exception:
        quality = 0.0
    return max(0.0, min(1.0, quality))


def _normalized_embedding(face: dict[str, Any]) -> np.ndarray | None:
    if not isinstance(face, dict):
        return None
    try:
        return normalize_embedding(parse_embedding(face.get("embedding") or []))
    except Exception:
        return None


def _build_similarity_adjacency(vectors: np.ndarray, *, min_similarity: float) -> np.ndarray:
    count = int(vectors.shape[0])
    adjacency = np.zeros((count, count), dtype=bool)
    for start in range(0, count, _SIMILARITY_BATCH_SIZE):
        stop = min(count, start + _SIMILARITY_BATCH_SIZE)
        similarity = vectors[start:stop] @ vectors.T
        adjacency[start:stop] = similarity >= float(min_similarity)
    return adjacency


def _dbscan_labels(adjacency: np.ndarray, *, min_samples: int) -> np.ndarray:
    count = int(adjacency.shape[0])
    labels = np.full(count, -1, dtype=np.int32)
    visited = np.zeros(count, dtype=bool)
    core_points = adjacency.sum(axis=1) >= max(1, int(min_samples))
    cluster_id = 0

    for index in range(count):
        if visited[index]:
            continue
        visited[index] = True
        if not bool(core_points[index]):
            continue
        labels[index] = cluster_id
        stack = list(np.flatnonzero(adjacency[index]))
        while stack:
            neighbor = int(stack.pop())
            if not visited[neighbor]:
                visited[neighbor] = True
                if bool(core_points[neighbor]):
                    stack.extend(int(item) for item in np.flatnonzero(adjacency[neighbor]))
            if labels[neighbor] < 0:
                labels[neighbor] = cluster_id
        cluster_id += 1
    return labels


def _medoid_index(vectors: np.ndarray) -> int:
    if int(vectors.shape[0]) == 1:
        return 0
    similarity = vectors @ vectors.T
    return int(np.argmax(similarity.sum(axis=1)))


def _distance_stats(vectors: np.ndarray, medoid: np.ndarray) -> dict[str, float]:
    similarity = np.clip(vectors @ medoid, -1.0, 1.0)
    distances = 1.0 - similarity
    return {
        "max_distance": float(np.max(distances)),
        "median_distance": float(np.median(distances)),
        "p90_distance": float(np.percentile(distances, 90)),
    }


def build_review_clusters(
    *,
    reviews: list[dict[str, Any]],
    faces_by_id: dict[str, dict[str, Any]],
    prototypes: dict[str, dict[str, Any]] | None,
    allowed_embedding_model_ids: set[str],
    eps: float = 0.28,
    min_samples: int = 3,
    top_k: int = 3,
    min_similarity: float = 0.72,
    min_margin: float = 0.015,
    min_face_quality: float = 0.20,
    min_sample_count: int = 2,
    min_consensus: float = 0.80,
    sample_face_limit: int = 4,
) -> list[dict[str, Any]]:
    pending_items: list[dict[str, Any]] = []
    for review in list(reviews or []):
        if not isinstance(review, dict):
            continue
        if str(review.get("status") or "").strip().lower() != "pending":
            continue
        face_id = str(review.get("face_id") or "").strip()
        if not face_id:
            continue
        face = faces_by_id.get(face_id)
        if not isinstance(face, dict):
            continue
        if str(face.get("person_id") or "").strip():
            continue
        if str(face.get("review_status") or "").strip().lower() in {"ignored", "rejected"}:
            continue
        if face_embedding_model(face) not in allowed_embedding_model_ids:
            continue
        vector = _normalized_embedding(face)
        if vector is None:
            continue
        pending_items.append(
            {
                "review_id": str(review.get("review_id") or "").strip(),
                "face_id": face_id,
                "vector": vector,
                "quality": _coerce_quality(face.get("quality")),
                "source_path": str(face.get("source_path") or "").strip(),
            }
        )

    if not pending_items:
        return []

    vectors = np.vstack([item["vector"] for item in pending_items]).astype(np.float32, copy=False)
    adjacency = _build_similarity_adjacency(vectors, min_similarity=max(-1.0, 1.0 - float(eps)))
    labels = _dbscan_labels(adjacency, min_samples=max(1, int(min_samples)))

    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, label in enumerate(labels.tolist()):
        if int(label) < 0:
            continue
        grouped.setdefault(int(label), []).append(pending_items[index])

    clusters: list[dict[str, Any]] = []
    for members in grouped.values():
        if len(members) < max(1, int(min_samples)):
            continue
        member_vectors = np.vstack([item["vector"] for item in members]).astype(np.float32, copy=False)
        medoid_offset = _medoid_index(member_vectors)
        medoid_member = members[medoid_offset]
        medoid_vector = member_vectors[medoid_offset]
        centroid = normalize_embedding(np.mean(member_vectors, axis=0))
        stats = _distance_stats(member_vectors, medoid_vector)

        ranked_offsets = np.argsort(np.clip(member_vectors @ medoid_vector, -1.0, 1.0))[::-1]
        sample_face_ids = [
            str(members[int(offset)]["face_id"]) for offset in ranked_offsets[: max(1, int(sample_face_limit))]
        ]

        source_paths = {str(item["source_path"]) for item in members if str(item["source_path"])}
        quality_values = [float(item["quality"]) for item in members]

        candidates: list[dict[str, Any]] = []
        suggested: dict[str, Any] | None = None
        suggested_margin = 0.0
        suggested_consensus = 0.0
        if prototypes:
            candidates = suggest_people_from_prototypes(
                query_embedding=centroid,
                prototypes=prototypes,
                top_k=max(3, int(top_k)),
                min_similarity=-1.0,
            )
            suggested, suggested_margin = choose_suggested_candidate(
                candidates=candidates,
                face_quality=float(np.median(quality_values)) if quality_values else 0.0,
                min_similarity=float(min_similarity),
                min_margin=float(min_margin),
                min_face_quality=float(min_face_quality),
                min_sample_count=max(1, int(min_sample_count)),
            )
            if suggested:
                top_person_ids: list[str] = []
                for item in members:
                    top_match = suggest_people_from_prototypes(
                        query_embedding=item["vector"],
                        prototypes=prototypes,
                        top_k=1,
                        min_similarity=-1.0,
                    )
                    if not top_match:
                        continue
                    top_person_id = str(top_match[0].get("person_id") or "").strip()
                    if top_person_id:
                        top_person_ids.append(top_person_id)
                suggested_person_id = str(suggested.get("person_id") or "").strip()
                if top_person_ids and suggested_person_id:
                    matching_count = Counter(top_person_ids).get(suggested_person_id, 0)
                    suggested_consensus = float(matching_count) / float(len(members))

        suggested_person_id = str((suggested or {}).get("person_id") or "").strip() or None
        suggested_score = float((suggested or {}).get("score") or 0.0) if suggested else None
        reviewable = bool(stats["p90_distance"] <= float(eps))
        suggested_confident = bool(reviewable and suggested_person_id and suggested_consensus >= float(min_consensus))

        clusters.append(
            {
                "cluster_id": str(medoid_member["face_id"]),
                "size": int(len(members)),
                "review_ids": [str(item["review_id"]) for item in members if str(item["review_id"])],
                "face_ids": [str(item["face_id"]) for item in members if str(item["face_id"])],
                "representative_face_id": str(medoid_member["face_id"]),
                "sample_face_ids": sample_face_ids,
                "source_count": int(len(source_paths)),
                "quality_median": float(np.median(quality_values)) if quality_values else 0.0,
                "quality_min": float(np.min(quality_values)) if quality_values else 0.0,
                "max_distance": float(stats["max_distance"]),
                "median_distance": float(stats["median_distance"]),
                "p90_distance": float(stats["p90_distance"]),
                "reviewable": reviewable,
                "suggested_person_id": suggested_person_id,
                "suggested_score": suggested_score,
                "suggested_margin": float(suggested_margin),
                "suggested_consensus": float(suggested_consensus),
                "suggested_confident": suggested_confident,
            }
        )

    clusters.sort(
        key=lambda row: (
            1 if bool(row.get("suggested_confident")) else 0,
            1 if bool(row.get("reviewable")) else 0,
            int(row.get("size") or 0),
        ),
        reverse=True,
    )
    return clusters
