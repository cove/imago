from cast.matching import (
    build_person_prototypes,
    parse_embedding,
    suggest_people,
    suggest_people_from_prototypes,
)


def test_parse_embedding_from_string():
    emb = parse_embedding("1.0, 0.0, -2.5")
    assert emb.shape == (3,)
    assert float(emb[0]) == 1.0
    assert float(emb[2]) == -2.5


def test_suggest_people_prefers_closest_person():
    faces = [
        {"person_id": "person_a", "embedding": [1.0, 0.0, 0.0]},
        {"person_id": "person_a", "embedding": [0.95, 0.1, 0.0]},
        {"person_id": "person_b", "embedding": [0.0, 1.0, 0.0]},
        {"person_id": "person_b", "embedding": [0.1, 0.9, 0.0]},
    ]
    results = suggest_people(query_embedding=[0.96, 0.05, 0.0], faces=faces, top_k=2)
    assert len(results) == 2
    assert results[0]["person_id"] == "person_a"
    assert results[0]["score"] > results[1]["score"]


def test_suggest_people_uses_best_exemplar_for_multimodal_person():
    faces = [
        {"person_id": "person_b", "embedding": [0.7, 0.7, 0.0], "quality": 0.9},
        {"person_id": "person_b", "embedding": [0.7, 0.7, 0.0], "quality": 0.9},
        {"person_id": "person_a", "embedding": [1.0, 0.0, 0.0], "quality": 0.9},
        {"person_id": "person_a", "embedding": [0.0, 1.0, 0.0], "quality": 0.9},
    ]
    results = suggest_people(query_embedding=[0.0, 1.0, 0.0], faces=faces, top_k=2)
    assert len(results) == 2
    assert results[0]["person_id"] == "person_a"


def test_suggest_people_from_prototypes_matches_direct_suggest():
    faces = [
        {"person_id": "a", "embedding": [1.0, 0.0, 0.0], "quality": 0.9},
        {"person_id": "a", "embedding": [0.9, 0.1, 0.0], "quality": 0.8},
        {"person_id": "b", "embedding": [0.0, 1.0, 0.0], "quality": 0.9},
    ]
    direct = suggest_people(query_embedding=[0.95, 0.05, 0.0], faces=faces, top_k=2)
    protos = build_person_prototypes(faces)
    from_proto = suggest_people_from_prototypes(
        query_embedding=[0.95, 0.05, 0.0],
        prototypes=protos,
        top_k=2,
    )
    assert [row["person_id"] for row in from_proto] == [
        row["person_id"] for row in direct
    ]


def test_build_person_prototypes_filters_faces_by_embedding_model():
    faces = [
        {
            "person_id": "new_person",
            "embedding": [1.0, 0.0, 0.0],
            "quality": 0.9,
            "metadata": {"embedding_model": "insightface.buffalo_l.arcface_512"},
        },
        {
            "person_id": "legacy_person",
            "embedding": [0.0, 1.0, 0.0],
            "quality": 0.9,
            "metadata": {"embedding_model": "legacy.arcface"},
        },
    ]

    protos = build_person_prototypes(
        faces,
        allowed_embedding_model_ids={"insightface.buffalo_l.arcface_512"},
    )

    assert set(protos) == {"new_person"}
