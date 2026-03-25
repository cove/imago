from cast.storage import TextFaceStore, face_review_status


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


def test_assign_face_can_mark_human_review_and_ignore(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    person = store.add_person(name="Audrey")
    face = store.add_face(
        embedding=[0.1, 0.2, 0.3],
        person_id=None,
        source_type="photo",
        source_path="photoalbums/a.jpg",
    )

    confirmed = store.assign_face(
        face["face_id"],
        person["person_id"],
        reviewed_by_human=True,
        review_status="confirmed",
    )
    assert confirmed["person_id"] == person["person_id"]
    assert confirmed["reviewed_by_human"] is True
    assert face_review_status(confirmed) == "confirmed"

    ignored = store.assign_face(
        face["face_id"],
        None,
        reviewed_by_human=True,
        review_status="ignored",
    )
    assert ignored["person_id"] is None
    assert ignored["reviewed_by_human"] is True
    assert face_review_status(ignored) == "ignored"


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


def test_update_person_name(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    person = store.add_person(name="Caria (Friend of Lynda)", aliases=["Caria"], notes="")
    person_id = str(person["person_id"])
    updated = store.update_person(person_id, display_name="Carla (Friend of Lynda)")

    assert updated["display_name"] == "Carla (Friend of Lynda)"
    loaded = store.get_person(person_id)
    assert loaded is not None
    assert loaded["display_name"] == "Carla (Friend of Lynda)"


def test_reset_pending_unknown_keeps_human_reviewed_ignored_faces(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    crops = store.root_dir / "crops"
    crops.mkdir(parents=True, exist_ok=True)

    ignored = store.add_face(
        embedding=[0.1, 0.2, 0.3],
        person_id=None,
        source_type="photo",
        crop_path="crops/ignored.jpg",
    )
    ignored = store.assign_face(
        ignored["face_id"],
        None,
        reviewed_by_human=True,
        review_status="ignored",
    )
    pending = store.add_face(
        embedding=[0.2, 0.3, 0.4],
        person_id=None,
        source_type="photo",
        crop_path="crops/pending.jpg",
    )
    store.add_review_item(
        face_id=pending["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )
    (crops / "ignored.jpg").write_bytes(b"ignored")
    (crops / "pending.jpg").write_bytes(b"pending")

    summary = store.reset_pending_unknown(remove_crops=True)
    assert summary["removed_faces"] == 1
    assert summary["removed_reviews"] == 1
    assert summary["removed_crops"] == 1

    faces = store.list_faces()
    assert len(faces) == 1
    assert faces[0]["face_id"] == ignored["face_id"]
    assert face_review_status(faces[0]) == "ignored"
    assert (crops / "ignored.jpg").exists()
    assert not (crops / "pending.jpg").exists()


def test_defer_review_item_keeps_pending_and_moves_item_to_back(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    first_face = store.add_face(
        embedding=[0.1, 0.2, 0.3],
        source_type="photo",
        source_path="photoalbums/first.jpg",
    )
    second_face = store.add_face(
        embedding=[0.2, 0.3, 0.4],
        source_type="photo",
        source_path="photoalbums/second.jpg",
    )
    first_review = store.add_review_item(
        face_id=first_face["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )
    second_review = store.add_review_item(
        face_id=second_face["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )

    deferred = store.defer_review_item(first_review["review_id"])
    assert deferred["status"] == "pending"
    assert deferred["skip_count"] == 1
    assert deferred["last_skipped_at"]

    reviews = store.list_review_items()
    assert [row["review_id"] for row in reviews] == [
        second_review["review_id"],
        first_review["review_id"],
    ]


def test_bulk_resolve_reviews_marks_faces_ignored(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    first_face = store.add_face(
        embedding=[0.1, 0.2, 0.3],
        source_type="photo",
        source_path="photoalbums/first.jpg",
    )
    second_face = store.add_face(
        embedding=[0.2, 0.3, 0.4],
        source_type="photo",
        source_path="photoalbums/second.jpg",
    )
    first_review = store.add_review_item(
        face_id=first_face["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )
    second_review = store.add_review_item(
        face_id=second_face["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )

    summary = store.bulk_resolve_reviews(
        review_ids=[first_review["review_id"], second_review["review_id"]],
        status="ignored",
    )

    assert summary == {"updated_reviews": 2, "updated_faces": 2}
    faces = {row["face_id"]: row for row in store.list_faces()}
    assert face_review_status(faces[first_face["face_id"]]) == "ignored"
    assert face_review_status(faces[second_face["face_id"]]) == "ignored"

    reviews = {row["review_id"]: row for row in store.list_review_items()}
    assert reviews[first_review["review_id"]]["status"] == "ignored"
    assert reviews[second_review["review_id"]]["status"] == "ignored"


def test_bulk_resolve_reviews_skips_and_moves_items_to_back(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    first_face = store.add_face(
        embedding=[0.1, 0.2, 0.3],
        source_type="photo",
        source_path="photoalbums/first.jpg",
    )
    second_face = store.add_face(
        embedding=[0.2, 0.3, 0.4],
        source_type="photo",
        source_path="photoalbums/second.jpg",
    )
    third_face = store.add_face(
        embedding=[0.3, 0.4, 0.5],
        source_type="photo",
        source_path="photoalbums/third.jpg",
    )
    first_review = store.add_review_item(
        face_id=first_face["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )
    second_review = store.add_review_item(
        face_id=second_face["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )
    third_review = store.add_review_item(
        face_id=third_face["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )

    summary = store.bulk_resolve_reviews(
        review_ids=[first_review["review_id"], third_review["review_id"]],
        status="skipped",
    )

    assert summary == {"updated_reviews": 2, "updated_faces": 0}
    reviews = store.list_review_items()
    assert [row["review_id"] for row in reviews] == [
        second_review["review_id"],
        first_review["review_id"],
        third_review["review_id"],
    ]
    assert reviews[1]["skip_count"] == 1
    assert reviews[2]["skip_count"] == 1
