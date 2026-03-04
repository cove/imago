from cast.matching import parse_embedding, suggest_people


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
