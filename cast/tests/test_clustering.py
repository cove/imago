from cast.clustering import build_review_clusters
from cast.ingest import CURRENT_FACE_EMBEDDING_MODEL
from cast.matching import build_person_prototypes


def test_build_review_clusters_groups_pending_faces_and_marks_confident_suggestion():
    person_faces = [
        {
            "face_id": "known-a",
            "person_id": "person-a",
            "embedding": [1.0, 0.0, 0.0],
            "quality": 0.95,
            "metadata": {"embedding_model": CURRENT_FACE_EMBEDDING_MODEL},
        },
        {
            "face_id": "known-b",
            "person_id": "person-a",
            "embedding": [0.99, 0.01, 0.0],
            "quality": 0.94,
            "metadata": {"embedding_model": CURRENT_FACE_EMBEDDING_MODEL},
        },
    ]
    pending_faces = {
        "pending-1": {
            "face_id": "pending-1",
            "person_id": None,
            "review_status": "",
            "source_path": "photo-1.jpg",
            "quality": 0.92,
            "embedding": [0.995, 0.005, 0.0],
            "metadata": {"embedding_model": CURRENT_FACE_EMBEDDING_MODEL},
        },
        "pending-2": {
            "face_id": "pending-2",
            "person_id": None,
            "review_status": "",
            "source_path": "photo-2.jpg",
            "quality": 0.90,
            "embedding": [0.992, 0.008, 0.0],
            "metadata": {"embedding_model": CURRENT_FACE_EMBEDDING_MODEL},
        },
        "pending-3": {
            "face_id": "pending-3",
            "person_id": None,
            "review_status": "",
            "source_path": "photo-3.jpg",
            "quality": 0.91,
            "embedding": [0.991, 0.009, 0.0],
            "metadata": {"embedding_model": CURRENT_FACE_EMBEDDING_MODEL},
        },
        "noise": {
            "face_id": "noise",
            "person_id": None,
            "review_status": "",
            "source_path": "photo-4.jpg",
            "quality": 0.91,
            "embedding": [0.0, 1.0, 0.0],
            "metadata": {"embedding_model": CURRENT_FACE_EMBEDDING_MODEL},
        },
    }
    reviews = [
        {"review_id": "review-1", "face_id": "pending-1", "status": "pending"},
        {"review_id": "review-2", "face_id": "pending-2", "status": "pending"},
        {"review_id": "review-3", "face_id": "pending-3", "status": "pending"},
        {"review_id": "review-4", "face_id": "noise", "status": "pending"},
    ]

    prototypes = build_person_prototypes(
        person_faces,
        allowed_embedding_model_ids={CURRENT_FACE_EMBEDDING_MODEL},
    )
    clusters = build_review_clusters(
        reviews=reviews,
        faces_by_id=pending_faces,
        prototypes=prototypes,
        allowed_embedding_model_ids={CURRENT_FACE_EMBEDDING_MODEL},
    )

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster["size"] == 3
    assert cluster["review_ids"] == ["review-1", "review-2", "review-3"]
    assert cluster["source_count"] == 3
    assert cluster["reviewable"] is True
    assert cluster["suggested_person_id"] == "person-a"
    assert cluster["suggested_confident"] is True
