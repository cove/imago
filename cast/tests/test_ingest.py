from pathlib import Path
from types import SimpleNamespace

import numpy as np

import cv2

from cast import ingest
from cast.ingest import FaceIngestor, crop_has_visual_detail
from cast.storage import TextFaceStore


def test_iter_photo_files_scans_view_dirs(tmp_path):
    root = tmp_path / "Photo Albums"
    keep_a = root / "1988_View"
    keep_b = root / "1992_View"
    skip = root / "1990_Archive"
    keep_a.mkdir(parents=True)
    keep_b.mkdir(parents=True)
    skip.mkdir(parents=True)

    (keep_a / "a.jpg").write_bytes(b"test")
    (keep_a / "b.jpeg").write_bytes(b"test")
    (keep_a / "ignore.png").write_bytes(b"test")
    (keep_b / "nested").mkdir()
    (keep_b / "nested" / "c.JPG").write_bytes(b"test")
    (skip / "d.jpg").write_bytes(b"test")

    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    ingestor = FaceIngestor(store)

    files = ingestor.iter_photo_files(
        photo_albums_root=root,
        view_glob="*_View",
        recursive=True,
    )
    names = sorted([Path(path).name for path in files])
    assert names == ["a.jpg", "b.jpeg", "c.JPG"]


def test_detect_handles_numpy_multirow_output(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    ingestor = FaceIngestor(store)
    ingestor._insightface = None

    class FakeCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.asarray(
                [
                    [10, 20, 40, 50],
                    [0, 0, 0, 10],  # invalid width should be filtered
                    [30, 35, 25, 25],
                ],
                dtype=np.int32,
            )

    ingestor._cascade = FakeCascade()  # type: ignore[assignment]
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    boxes = ingestor._detect(image, min_size=20)
    assert boxes == [(10, 20, 40, 50), (30, 35, 25, 25)]


def test_crop_has_visual_detail_rejects_flat_color():
    flat = np.zeros((80, 80, 3), dtype=np.uint8)
    flat[:, :] = (255, 0, 0)
    assert crop_has_visual_detail(flat) is False

    textured = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.circle(textured, (25, 30), 12, (200, 200, 200), -1)
    cv2.circle(textured, (55, 35), 10, (40, 40, 40), -1)
    cv2.line(textured, (20, 60), (60, 58), (220, 220, 220), 2)
    assert crop_has_visual_detail(textured) is True


def test_ingest_photo_skips_flat_false_positive_detection(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    ingestor = FaceIngestor(store)
    ingestor._insightface = None

    class FakeCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.asarray([[10, 10, 50, 50]], dtype=np.int32)

    ingestor._cascade = FakeCascade()  # type: ignore[assignment]

    image_path = tmp_path / "solid_blue.jpg"
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)
    cv2.imwrite(str(image_path), image)

    created = ingestor.ingest_photo(image_path=image_path, max_faces=5)
    assert created == []


def test_detect_handles_opencv_cascade_error(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    ingestor = FaceIngestor(store)
    ingestor._insightface = None

    class BrokenCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            raise cv2.error("OpenCV(4.10.0) ... error: (-215:Assertion failed) 0 <= scaleIdx ...")

    ingestor._cascade = BrokenCascade()  # type: ignore[assignment]
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    boxes = ingestor._detect(image, min_size=40)
    assert boxes == []


def test_ingest_photo_rejects_directory_path(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    ingestor = FaceIngestor(store)

    folder = tmp_path / "not_a_photo"
    folder.mkdir()

    try:
        ingestor.ingest_photo(image_path=folder)
        assert False, "Expected IsADirectoryError"
    except IsADirectoryError:
        pass


def test_ingest_photo_requires_primary_model_when_configured(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    ingestor = FaceIngestor(store, require_primary_model=True)
    ingestor._insightface = None

    image_path = tmp_path / "photo.jpg"
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (90, 90), (220, 220, 220), -1)
    cv2.imwrite(str(image_path), image)

    try:
        ingestor.ingest_photo(image_path=image_path, max_faces=5)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "InsightFace buffalo_l is unavailable" in str(exc)


def test_resolve_haarcascade_dir_falls_back_to_cv2_data_dir_neighbor(tmp_path, monkeypatch):
    cv2_dir = tmp_path / "cv2"
    data_dir = cv2_dir / "data"
    data_dir.mkdir(parents=True)

    monkeypatch.setattr(
        ingest,
        "cv2",
        SimpleNamespace(__file__=str(cv2_dir / "__init__.py"), data=SimpleNamespace()),
    )

    assert ingest._resolve_haarcascade_dir() == data_dir
