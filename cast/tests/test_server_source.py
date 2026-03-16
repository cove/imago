import json
import threading
import time
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from cast.server import CastHTTPServer
from cast.storage import TextFaceStore


def _write_test_video(tmp_path: Path) -> tuple[Path, float]:
    size = (320, 240)
    fps = 10.0
    codecs = [
        ("sample_mjpg.avi", "MJPG"),
        ("sample_xvid.avi", "XVID"),
        ("sample_mp4v.mp4", "mp4v"),
    ]
    for filename, fourcc_name in codecs:
        path = tmp_path / filename
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc_name), fps, size
        )
        if not writer.isOpened():
            writer.release()
            continue
        try:
            for index in range(12):
                frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                frame[:, :] = (index * 8, 18 + index * 6, 22 + index * 4)
                cv2.rectangle(frame, (90, 70), (170, 170), (210, 210, 210), -1)
                writer.write(frame)
            return path, fps
        finally:
            writer.release()
    raise RuntimeError("Unable to create test video file with available OpenCV codecs.")


def test_source_image_endpoint_and_state_source_url(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    source_image = tmp_path / "photo.jpg"
    image = np.zeros((220, 320, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 60), (180, 180), (180, 180, 180), -1)
    cv2.imwrite(str(source_image), image)

    face = store.add_face(
        embedding=[0.1, 0.2, 0.3],
        source_type="photo",
        source_path=str(source_image),
        bbox=[80, 60, 100, 120],
    )
    face_id = str(face["face_id"])

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/state", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload.get("ok") is True
        faces = payload.get("faces") or []
        row = next(item for item in faces if str(item.get("face_id")) == face_id)
        assert str(row.get("source_url", "")).endswith(f"/api/faces/{face_id}/source")

        with urlopen(
            f"http://127.0.0.1:{port}/api/faces/{face_id}/source?highlight=1",
            timeout=10,
        ) as response:
            data = response.read()
            ctype = str(response.headers.get("Content-Type", ""))
        assert ctype.startswith("image/jpeg")
        assert len(data) > 1000
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_source_video_endpoint_and_state_source_url_for_vhs(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    source_video, fps = _write_test_video(tmp_path)
    frame_index = 5
    timestamp_seconds = float(frame_index) / float(fps)
    timestamp = f"00:00:00.{int(round(timestamp_seconds * 1000.0)):03d}"

    face = store.add_face(
        embedding=[0.2, 0.3, 0.4],
        source_type="vhs",
        source_path=str(source_video),
        timestamp=timestamp,
        bbox=[90, 70, 80, 100],
        metadata={"ingest": "vhs", "frame_index": frame_index, "fps": fps},
    )
    face_id = str(face["face_id"])

    server = CastHTTPServer("127.0.0.1", 0, store)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/state", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload.get("ok") is True
        faces = payload.get("faces") or []
        row = next(item for item in faces if str(item.get("face_id")) == face_id)
        assert str(row.get("source_url", "")).endswith(f"/api/faces/{face_id}/source")
        assert row.get("source_is_image") is False

        with urlopen(
            f"http://127.0.0.1:{port}/api/faces/{face_id}/source?highlight=0",
            timeout=10,
        ) as response:
            plain_data = response.read()
            plain_ctype = str(response.headers.get("Content-Type", ""))
        with urlopen(
            f"http://127.0.0.1:{port}/api/faces/{face_id}/source?highlight=1",
            timeout=10,
        ) as response:
            highlighted_data = response.read()
            highlighted_ctype = str(response.headers.get("Content-Type", ""))

        plain_img = cv2.imdecode(
            np.frombuffer(plain_data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        highlighted_img = cv2.imdecode(
            np.frombuffer(highlighted_data, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        assert plain_ctype.startswith("image/jpeg")
        assert highlighted_ctype.startswith("image/jpeg")
        assert plain_img is not None
        assert highlighted_img is not None
        assert plain_img.shape == highlighted_img.shape
        assert int(np.sum(cv2.absdiff(plain_img, highlighted_img))) > 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
