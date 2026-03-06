from cast.server import choose_suggested_candidate


def test_choose_suggested_candidate_rejects_low_margin():
    candidates = [
        {"person_id": "a", "score": 0.91, "sample_count": 8},
        {"person_id": "b", "score": 0.905, "sample_count": 8},
    ]
    suggested, margin = choose_suggested_candidate(
        candidates=candidates,
        face_quality=0.6,
        min_similarity=0.72,
        min_margin=0.015,
        min_face_quality=0.2,
        min_sample_count=2,
    )
    assert suggested is None
    assert 0.0 <= float(margin) < 0.015


def test_choose_suggested_candidate_rejects_low_quality():
    candidates = [
        {"person_id": "a", "score": 0.93, "sample_count": 8},
        {"person_id": "b", "score": 0.84, "sample_count": 8},
    ]
    suggested, margin = choose_suggested_candidate(
        candidates=candidates,
        face_quality=0.12,
        min_similarity=0.72,
        min_margin=0.015,
        min_face_quality=0.2,
        min_sample_count=2,
    )
    assert suggested is None
    assert float(margin) >= 0.015


def test_choose_suggested_candidate_accepts_confident_match():
    candidates = [
        {"person_id": "a", "score": 0.94, "sample_count": 5},
        {"person_id": "b", "score": 0.88, "sample_count": 10},
    ]
    suggested, margin = choose_suggested_candidate(
        candidates=candidates,
        face_quality=0.42,
        min_similarity=0.72,
        min_margin=0.015,
        min_face_quality=0.2,
        min_sample_count=2,
    )
    assert suggested is not None
    assert str(suggested.get("person_id")) == "a"
    assert float(margin) >= 0.015
