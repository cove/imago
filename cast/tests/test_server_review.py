import json
import io
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cast.server as cast_server
from cast.server import CastHTTPServer
from cast.storage import TextFaceStore


def _post_json(url: str, payload: dict) -> tuple[int, dict]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            return int(response.status), json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return int(exc.code), json.loads(exc.read().decode("utf-8"))


def _get_json(url: str) -> tuple[int, dict]:
    with urlopen(url, timeout=10) as response:
        return int(response.status), json.loads(response.read().decode("utf-8"))


def test_wait_for_photoalbums_processing_lock_waits_for_release(tmp_path):
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"jpg")
    lock_path = cast_server._photoalbums_processing_lock_path(image_path)
    lock_path.write_text("{}", encoding="utf-8")

    def release_lock():
        time.sleep(0.1)
        lock_path.unlink()

    thread = threading.Thread(target=release_lock, daemon=True)
    thread.start()
    try:
        waited = cast_server._wait_for_photoalbums_processing_lock(
            image_path,
            timeout_seconds=2.0,
            poll_seconds=0.02,
        )
        assert waited is True
    finally:
        thread.join(timeout=2)


def test_review_skip_keeps_item_pending_and_moves_it_to_end(tmp_path):
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

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{port}/api/review/resolve",
            {"review_id": first_review["review_id"], "status": "skipped"},
        )
        assert status == 200
        assert payload["ok"] is True
        assert payload["review"]["status"] == "pending"
        assert payload["review"]["skip_count"] == 1

        with urlopen(f"http://127.0.0.1:{port}/api/review?status=pending", timeout=10) as response:
            reviews_payload = json.loads(response.read().decode("utf-8"))
        assert [row["review_id"] for row in reviews_payload["reviews"]] == [
            second_review["review_id"],
            first_review["review_id"],
        ]

        with urlopen(f"http://127.0.0.1:{port}/api/state", timeout=10) as response:
            state_payload = json.loads(response.read().decode("utf-8"))
        assert state_payload["counts"]["pending_reviews"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_server_disables_caching_for_ui_and_state(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        with urlopen(f"http://127.0.0.1:{port}/", timeout=10) as response:
            cache_control_root = str(response.headers.get("Cache-Control", ""))
        with urlopen(f"http://127.0.0.1:{port}/api/state", timeout=10) as response:
            cache_control_state = str(response.headers.get("Cache-Control", ""))

        assert "no-store" in cache_control_root
        assert "no-store" in cache_control_state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_bulk_review_resolve_ignores_multiple_pending_items(tmp_path):
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

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{port}/api/review/bulk_resolve",
            {
                "review_ids": [first_review["review_id"], second_review["review_id"]],
                "status": "ignored",
            },
        )
        assert status == 200
        assert payload["ok"] is True
        assert payload["updated_reviews"] == 2
        assert payload["updated_faces"] == 2

        faces = {row["face_id"]: row for row in store.list_faces()}
        assert faces[first_face["face_id"]]["review_status"] == "ignored"
        assert faces[second_face["face_id"]]["review_status"] == "ignored"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_bulk_review_assign_creates_person_once_and_assigns_many(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    first_face = store.add_face(embedding=[0.1, 0.2, 0.3], source_type="photo")
    second_face = store.add_face(embedding=[0.2, 0.3, 0.4], source_type="photo")
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

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{port}/api/review/bulk_assign",
            {
                "review_ids": [first_review["review_id"], second_review["review_id"]],
                "display_name": "Audrey",
            },
        )
        assert status == 200
        assert payload["ok"] is True
        assert payload["created_person"] is True
        assert payload["updated_reviews"] == 2
        assert payload["updated_faces"] == 2

        people = store.list_people()
        assert len(people) == 1
        person_id = str(people[0]["person_id"])
        assert people[0]["display_name"] == "Audrey"

        faces = {row["face_id"]: row for row in store.list_faces()}
        assert faces[first_face["face_id"]]["person_id"] == person_id
        assert faces[second_face["face_id"]]["person_id"] == person_id
        assert faces[first_face["face_id"]]["review_status"] == "confirmed"
        assert faces[second_face["face_id"]]["review_status"] == "confirmed"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_review_clusters_endpoint_groups_pending_faces_and_exposes_bulk_ids(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    person = store.add_person(name="Audrey")
    store.add_face(
        embedding=[1.0, 0.0, 0.0],
        person_id=person["person_id"],
        source_type="photo",
        quality=0.95,
        metadata={"embedding_model": cast_server.CURRENT_FACE_EMBEDDING_MODEL},
    )
    store.add_face(
        embedding=[0.99, 0.01, 0.0],
        person_id=person["person_id"],
        source_type="photo",
        quality=0.94,
        metadata={"embedding_model": cast_server.CURRENT_FACE_EMBEDDING_MODEL},
    )
    pending_faces = [
        store.add_face(
            embedding=[0.995, 0.005, 0.0],
            source_type="photo",
            source_path=f"photoalbums/pending-{index}.jpg",
            quality=0.93,
            metadata={"embedding_model": cast_server.CURRENT_FACE_EMBEDDING_MODEL},
        )
        for index in range(3)
    ]
    for face in pending_faces:
        store.add_review_item(
            face_id=face["face_id"],
            candidates=[],
            suggested_person_id=None,
            suggested_score=None,
            status="pending",
        )

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        status, payload = _get_json(f"http://127.0.0.1:{port}/api/review/clusters")
        assert status == 200
        assert payload["ok"] is True
        assert payload["counts"]["clusters"] == 1
        assert payload["counts"]["clustered_reviews"] == 3
        cluster = payload["clusters"][0]
        assert cluster["size"] == 3
        assert cluster["suggested_person_id"] == person["person_id"]
        assert cluster["suggested_person_name"] == "Audrey"
        assert cluster["suggested_confident"] is True
        assert len(cluster["review_ids"]) == 3
        assert len(cluster["sample_faces"]) >= 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_bulk_review_assign_reuses_existing_alias_match(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    person = store.add_person(name="Audrey", aliases=["Auds"])
    face = store.add_face(embedding=[0.1, 0.2, 0.3], source_type="photo")
    review = store.add_review_item(
        face_id=face["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{port}/api/review/bulk_assign",
            {
                "review_ids": [review["review_id"]],
                "display_name": "auds",
            },
        )
        assert status == 200
        assert payload["ok"] is True
        assert payload["created_person"] is False
        assert str(payload["person"]["person_id"]) == str(person["person_id"])
        assert len(store.list_people()) == 1

        refreshed_face = store.list_faces()[0]
        assert refreshed_face["person_id"] == person["person_id"]
        assert refreshed_face["review_status"] == "confirmed"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_state_limits_pending_and_unknown_payload_rows(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    for index in range(5):
        face = store.add_face(
            embedding=[0.1, 0.2, 0.3],
            source_type="photo",
            source_path=f"photoalbums/{index}.jpg",
        )
        store.add_review_item(
            face_id=face["face_id"],
            candidates=[],
            suggested_person_id=None,
            suggested_score=None,
            status="pending",
        )

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        with urlopen(
            f"http://127.0.0.1:{port}/api/state?pending_limit=2&unknown_limit=3",
            timeout=10,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert payload["counts"]["pending_reviews"] == 5
        assert payload["counts"]["unknown_faces"] == 5
        assert len(payload["pending_reviews"]) == 2
        assert len(payload["unknown_faces"]) == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_state_reports_runtime_and_legacy_face_count(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    store.add_face(
        embedding=[0.1, 0.2, 0.3],
        source_type="photo",
        source_path="photoalbums/legacy.jpg",
    )
    store.add_face(
        embedding=[0.2, 0.3, 0.4],
        source_type="photo",
        source_path="photoalbums/current.jpg",
        metadata={
            "detector_model": "insightface.buffalo_l.detector",
            "embedding_model": "insightface.buffalo_l.arcface_512",
        },
    )

    server = CastHTTPServer("127.0.0.1", 0, store)
    server.ingestor._insightface = None
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/state", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert payload["counts"]["legacy_faces"] == 1
        assert payload["runtime"]["primary_required"] is True
        assert payload["runtime"]["primary_available"] is False
        assert payload["runtime"]["can_ingest"] is False
        assert payload["runtime"]["active_detector_model"] == "opencv.haar_frontalface_default"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_server_blocks_ingest_when_primary_model_is_unavailable(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"not used")

    server = CastHTTPServer("127.0.0.1", 0, store)
    server.ingestor._insightface = None
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{port}/api/ingest/photo",
            {"image_path": str(image_path), "auto_queue": True},
        )
        assert status == 400
        assert payload["ok"] is False
        assert "InsightFace buffalo_l is unavailable" in payload["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_photo_scan_rescan_returns_removed_counts(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    server = CastHTTPServer("127.0.0.1", 0, store)

    def fake_ingest_photo_album_views(**kwargs):
        assert kwargs["rescan_existing"] is True
        return {
            "photo_files_scanned": 3,
            "view_files_scanned": 2,
            "archive_scan_files_scanned": 1,
            "faces_created": 5,
            "faces": [],
            "per_photo": [],
            "photo_albums_root": str(tmp_path),
            "view_glob": "Family*_Pages",
            "rescan_existing": True,
            "removed_faces": 4,
            "removed_reviews": 6,
            "removed_crops": 4,
        }

    server.ingestor.ingest_photo_album_views = fake_ingest_photo_album_views  # type: ignore[method-assign]
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{port}/api/ingest/photos/scan",
            {
                "photo_albums_root": str(tmp_path),
                "view_glob": "Family*_Pages",
                "auto_queue": False,
                "rescan_existing": True,
            },
        )
        assert status == 201
        assert payload["ok"] is True
        assert payload["rescan_existing"] is True
        assert payload["view_files_scanned"] == 2
        assert payload["archive_scan_files_scanned"] == 1
        assert payload["removed_faces"] == 4
        assert payload["removed_reviews"] == 6
        assert payload["removed_crops"] == 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_review_accept_keeps_item_pending_when_xmp_write_fails(tmp_path, monkeypatch):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    person = store.add_person(name="Audrey")
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"jpg")
    face = store.add_face(
        embedding=[0.1, 0.2, 0.3],
        source_type="photo",
        source_path=str(image_path),
        bbox=[10, 20, 30, 40],
    )
    review = store.add_review_item(
        face_id=face["face_id"],
        candidates=[],
        suggested_person_id=person["person_id"],
        suggested_score=0.95,
        status="pending",
    )

    stderr_buffer = io.StringIO()
    monkeypatch.setattr(cast_server.sys, "stderr", stderr_buffer)
    monkeypatch.setattr(cast_server, "read_person_in_image", lambda path: [])
    monkeypatch.setattr(cast_server, "read_xmp_description", lambda path: "")

    def raise_merge(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(image_path.with_suffix(".xmp")))

    monkeypatch.setattr(cast_server, "merge_persons_xmp", raise_merge)

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{port}/api/review/resolve",
            {
                "review_id": review["review_id"],
                "status": "accepted",
                "person_id": person["person_id"],
            },
        )
        assert status == 500
        assert payload["ok"] is False
        assert "XMP write-back failed for review" in payload["error"]

        refreshed_review = store.list_review_items()[0]
        refreshed_face = store.list_faces()[0]
        assert refreshed_review["status"] == "pending"
        assert refreshed_review["decided_person_id"] is None
        assert refreshed_face["person_id"] is None
        assert refreshed_face["review_status"] == ""

        log_text = stderr_buffer.getvalue()
        assert '"event": "xmp_write_back_failed"' in log_text
        assert str(review["review_id"]) in log_text
        assert str(face["face_id"]) in log_text
        assert str(person["person_id"]) in log_text
        assert '"source_path":' in log_text
        assert '"xmp_path":' in log_text
        assert "photo.jpg" in log_text
        assert "photo.xmp" in log_text
        assert "PermissionError" in log_text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_bulk_review_assign_stops_when_xmp_write_fails(tmp_path, monkeypatch):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    person = store.add_person(name="Audrey")
    good_face = store.add_face(
        embedding=[0.1, 0.2, 0.3],
        source_type="photo",
    )
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"jpg")
    failing_face = store.add_face(
        embedding=[0.2, 0.3, 0.4],
        source_type="photo",
        source_path=str(image_path),
        bbox=[10, 20, 30, 40],
    )
    good_review = store.add_review_item(
        face_id=good_face["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )
    failing_review = store.add_review_item(
        face_id=failing_face["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )

    monkeypatch.setattr(cast_server, "read_person_in_image", lambda path: [])

    def raise_merge(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(image_path.with_suffix(".xmp")))

    monkeypatch.setattr(cast_server, "merge_persons_xmp", raise_merge)

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{port}/api/review/bulk_assign",
            {
                "review_ids": [good_review["review_id"], failing_review["review_id"]],
                "person_id": person["person_id"],
            },
        )
        assert status == 500
        assert payload["ok"] is False
        assert "Bulk assign stopped after 1 review(s)" in payload["error"]
        assert "XMP write-back failed for review" in payload["error"]

        reviews = {row["review_id"]: row for row in store.list_review_items()}
        assert reviews[good_review["review_id"]]["status"] == "accepted"
        assert reviews[failing_review["review_id"]]["status"] == "pending"

        faces = {row["face_id"]: row for row in store.list_faces()}
        assert faces[good_face["face_id"]]["person_id"] == person["person_id"]
        assert faces[failing_face["face_id"]]["person_id"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_review_accept_waits_for_photoalbums_lock_before_xmp_write(tmp_path, monkeypatch):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    person = store.add_person(name="Audrey")
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"jpg")
    face = store.add_face(
        embedding=[0.1, 0.2, 0.3],
        source_type="photo",
        source_path=str(image_path),
        bbox=[10, 20, 30, 40],
    )
    review = store.add_review_item(
        face_id=face["face_id"],
        candidates=[],
        suggested_person_id=person["person_id"],
        suggested_score=0.95,
        status="pending",
    )

    waited_paths = []

    def fake_wait(path, **kwargs):
        waited_paths.append((path, kwargs))
        return True

    read_calls = {"count": 0}

    def fake_read_person_in_image(path):
        read_calls["count"] += 1
        if read_calls["count"] == 1:
            return []
        return ["Audrey"]

    monkeypatch.setattr(cast_server, "_wait_for_photoalbums_processing_lock", fake_wait)
    monkeypatch.setattr(cast_server, "read_person_in_image", fake_read_person_in_image)
    monkeypatch.setattr(cast_server, "read_xmp_description", lambda path: "")
    monkeypatch.setattr(cast_server, "merge_persons_xmp", lambda *args, **kwargs: image_path.with_suffix(".xmp"))

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{port}/api/review/resolve",
            {
                "review_id": review["review_id"],
                "status": "accepted",
                "person_id": person["person_id"],
            },
        )
        assert status == 200
        assert payload["ok"] is True
        assert payload["review"]["status"] == "accepted"
        assert waited_paths
        assert waited_paths[0][0] == image_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_review_accept_keeps_item_pending_when_person_in_image_verification_fails(tmp_path, monkeypatch):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    person = store.add_person(name="Audrey")
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"jpg")
    face = store.add_face(
        embedding=[0.1, 0.2, 0.3],
        source_type="photo",
        source_path=str(image_path),
        bbox=[10, 20, 30, 40],
    )
    review = store.add_review_item(
        face_id=face["face_id"],
        candidates=[],
        suggested_person_id=person["person_id"],
        suggested_score=0.95,
        status="pending",
    )

    stderr_buffer = io.StringIO()
    monkeypatch.setattr(cast_server.sys, "stderr", stderr_buffer)
    read_calls = {"count": 0}

    def fake_read_person_in_image(path):
        read_calls["count"] += 1
        if read_calls["count"] == 1:
            return []
        return ["Someone Else"]

    monkeypatch.setattr(cast_server, "read_person_in_image", fake_read_person_in_image)
    monkeypatch.setattr(cast_server, "read_xmp_description", lambda path: "")
    monkeypatch.setattr(cast_server, "merge_persons_xmp", lambda *args, **kwargs: image_path.with_suffix(".xmp"))

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        status, payload = _post_json(
            f"http://127.0.0.1:{port}/api/review/resolve",
            {
                "review_id": review["review_id"],
                "status": "accepted",
                "person_id": person["person_id"],
            },
        )
        assert status == 500
        assert payload["ok"] is False
        assert "Assigned person was not written to Iptc4xmpExt:PersonInImage" in payload["error"]

        refreshed_review = store.list_review_items()[0]
        refreshed_face = store.list_faces()[0]
        assert refreshed_review["status"] == "pending"
        assert refreshed_face["person_id"] is None

        log_text = stderr_buffer.getvalue()
        assert '"event": "xmp_write_back_verification_failed"' in log_text
        assert '"verified_person_in_image": ["Someone Else"]' in log_text
        assert str(review["review_id"]) in log_text
        assert str(face["face_id"]) in log_text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

