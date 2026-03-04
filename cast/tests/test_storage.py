from cast.storage import TextFaceStore


def test_store_round_trip(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    person = store.add_person(name="Jim Bennett", aliases=["Jimmy"], notes="Test person")
    assert person["display_name"] == "Jim Bennett"

    face = store.add_face(
        embedding=[0.1, 0.2, 0.3],
        person_id=None,
        source_type="photo",
        source_path="photoalbums/1992_book_01/p01.jpg",
    )
    assert face["person_id"] is None
    updated = store.update_face(face["face_id"], crop_path="crops/example.jpg")
    assert updated["crop_path"] == "crops/example.jpg"

    assigned = store.assign_face(face["face_id"], person["person_id"])
    assert assigned["person_id"] == person["person_id"]

    review = store.add_review_item(
        face_id=face["face_id"],
        candidates=[{"person_id": person["person_id"], "score": 0.92, "sample_count": 1}],
        suggested_person_id=person["person_id"],
        suggested_score=0.92,
    )
    assert review["status"] == "pending"

    resolved = store.resolve_review_item(
        review_id=review["review_id"],
        status="accepted",
        decided_person_id=person["person_id"],
    )
    assert resolved["status"] == "accepted"
    assert resolved["decided_person_id"] == person["person_id"]


def test_reset_pending_unknown_keeps_assigned_faces(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    crops = store.root_dir / "crops"
    crops.mkdir(parents=True, exist_ok=True)

    person = store.add_person(name="Audrey")
    known = store.add_face(
        embedding=[0.2, 0.3, 0.4],
        person_id=person["person_id"],
        source_type="photo",
        crop_path="crops/known.jpg",
    )
    unknown = store.add_face(
        embedding=[0.1, 0.2, 0.3],
        person_id=None,
        source_type="photo",
        crop_path="crops/unknown.jpg",
    )
    (crops / "known.jpg").write_bytes(b"known")
    (crops / "unknown.jpg").write_bytes(b"unknown")

    store.add_review_item(
        face_id=unknown["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )
    accepted = store.add_review_item(
        face_id=known["face_id"],
        candidates=[],
        suggested_person_id=person["person_id"],
        suggested_score=0.9,
        status="accepted",
    )

    summary = store.reset_pending_unknown(remove_crops=True)
    assert summary["removed_faces"] == 1
    assert summary["removed_reviews"] == 1
    assert summary["removed_crops"] == 1

    faces = store.list_faces()
    assert len(faces) == 1
    assert faces[0]["face_id"] == known["face_id"]

    reviews = store.list_review_items()
    assert len(reviews) == 1
    assert reviews[0]["review_id"] == accepted["review_id"]
    assert (crops / "known.jpg").exists()
    assert not (crops / "unknown.jpg").exists()
