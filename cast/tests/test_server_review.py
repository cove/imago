import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

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

        with urlopen(
            f"http://127.0.0.1:{port}/api/review?status=pending", timeout=10
        ) as response:
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
        assert (
            payload["runtime"]["active_detector_model"]
            == "opencv.haar_frontalface_default"
        )
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
