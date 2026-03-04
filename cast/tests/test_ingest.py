from pathlib import Path

import numpy as np

import cv2

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
    ingestor._yunet = None

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

    ingestor._cascade = FakeCascade()
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

    class FakeCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.asarray([[10, 10, 50, 50]], dtype=np.int32)

    ingestor._cascade = FakeCascade()

    image_path = tmp_path / "solid_blue.jpg"
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)
    cv2.imwrite(str(image_path), image)

    created = ingestor.ingest_photo(image_path=image_path, max_faces=5)
    assert created == []


def test_ingest_photo_skips_torso_like_false_positive_detection(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    ingestor = FaceIngestor(store)

    class FakeCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.asarray([[20, 20, 120, 120]], dtype=np.int32)

    ingestor._cascade = FakeCascade()

    image_path = tmp_path / "torso_like.jpg"
    image = np.zeros((220, 180, 3), dtype=np.uint8)
    image[:90, :] = (130, 130, 130)
    image[90:, :] = (170, 95, 55)
    cv2.imwrite(str(image_path), image)

    created = ingestor.ingest_photo(image_path=image_path, max_faces=3)
    assert created == []


def test_yunet_landmark_geometry_rejects_nonsense(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    ingestor = FaceIngestor(store)

    box = (10, 10, 80, 80)
    row = np.asarray(
        [
            10, 10, 80, 80,  # bbox
            55, 30,          # left eye (bad: right of right eye)
            30, 30,          # right eye
            42, 35,          # nose
            25, 55,          # mouth left
            58, 56,          # mouth right
            0.99,            # score
        ],
        dtype=np.float32,
    )
    assert ingestor._yunet_landmarks_plausible(row, box) is False


def test_looks_like_artwork_face_flags_letter_shape(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    ingestor = FaceIngestor(store)

    image = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.circle(image, (48, 48), 24, (255, 255, 255), 5)
    assert ingestor.looks_like_artwork_face(image) is True


def test_detect_handles_opencv_cascade_error(tmp_path):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()
    ingestor = FaceIngestor(store)
    ingestor._yunet = None

    class BrokenCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            raise cv2.error(
                "OpenCV(4.10.0) ... error: (-215:Assertion failed) 0 <= scaleIdx ..."
            )

    ingestor._cascade = BrokenCascade()
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
