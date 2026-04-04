from pathlib import Path
from types import SimpleNamespace
import io
import contextlib

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


def test_ingest_photo_album_views_rescan_removes_matching_faces_and_reviews(tmp_path):
    root = tmp_path / "Photo Albums"
    family_view = root / "Family_View"
    family_archive = root / "Family_Archive"
    other_view = root / "Other_View"
    family_view.mkdir(parents=True)
    family_archive.mkdir(parents=True)
    other_view.mkdir(parents=True)
    family_a = family_view / "a.jpg"
    family_b = family_view / "b.jpg"
    family_scan = family_archive / "a.tif"
    other_photo = other_view / "other.jpg"
    family_a.write_bytes(b"a")
    family_b.write_bytes(b"b")
    family_scan.write_bytes(b"scan")
    other_photo.write_bytes(b"c")

    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    ingestor = FaceIngestor(store)

    removed_face = store.add_face(
        embedding=[0.1, 0.2, 0.3],
        source_type="photo",
        source_path=str(family_a.resolve()),
        crop_path="crops/removed.jpg",
    )
    removed_scan_face = store.add_face(
        embedding=[0.4, 0.5, 0.6],
        source_type="photo",
        source_path=str(family_scan.resolve()),
        crop_path="crops/removed-scan.jpg",
    )
    kept_face = store.add_face(
        embedding=[0.2, 0.3, 0.4],
        source_type="photo",
        source_path=str(other_photo.resolve()),
        crop_path="crops/kept.jpg",
    )
    (store.root_dir / "crops").mkdir(parents=True, exist_ok=True)
    (store.root_dir / "crops" / "removed.jpg").write_bytes(b"removed")
    (store.root_dir / "crops" / "removed-scan.jpg").write_bytes(b"removed-scan")
    (store.root_dir / "crops" / "kept.jpg").write_bytes(b"kept")
    store.add_review_item(
        face_id=removed_face["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )
    store.add_review_item(
        face_id=removed_scan_face["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )
    store.add_review_item(
        face_id=kept_face["face_id"],
        candidates=[],
        suggested_person_id=None,
        suggested_score=None,
        status="pending",
    )

    ingested_paths = []

    def fake_ingest_photo(*, image_path, source_path=None, min_size=40, max_faces=50):
        ingested_paths.append(str(image_path))
        return []

    ingestor.ingest_photo = fake_ingest_photo  # type: ignore[method-assign]

    result = ingestor.ingest_photo_album_views(
        photo_albums_root=root,
        view_glob="Family_View",
        recursive=True,
        max_files=0,
        rescan_existing=True,
    )

    assert result["photo_files_scanned"] == 3
    assert result["view_files_scanned"] == 2
    assert result["archive_scan_files_scanned"] == 1
    assert result["removed_faces"] == 2
    assert result["removed_reviews"] == 2
    assert result["removed_crops"] == 2
    assert ingested_paths == [
        str(family_a.resolve()),
        str(family_b.resolve()),
        str(family_scan.resolve()),
    ]
    assert (store.root_dir / "crops" / "removed.jpg").exists() is False
    assert (store.root_dir / "crops" / "removed-scan.jpg").exists() is False
    assert (store.root_dir / "crops" / "kept.jpg").exists() is True
    assert [row["face_id"] for row in store.list_faces()] == [kept_face["face_id"]]
    assert [row["face_id"] for row in store.list_review_items()] == [kept_face["face_id"]]


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


def test_get_insightface_app_suppresses_startup_noise(monkeypatch):
    class FakeApp:
        def prepare(self, **_kwargs):
            print("Applied providers: ['CPUExecutionProvider']")
            print("set det-size: (640, 640)")

    def fake_init():
        print("find model: fake-model.onnx recognition")
        return FakeApp()

    monkeypatch.setattr(ingest, "_insightface_app", None)
    monkeypatch.setattr(ingest, "_insightface_app_error", "")
    monkeypatch.setattr(ingest, "_init_insightface_app", fake_init)

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        app = ingest._get_insightface_app()

    assert app is not None
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == ""


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
