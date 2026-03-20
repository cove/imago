from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cast.storage import TextFaceStore
from photoalbums.lib.ai_people import CastPeopleMatcher
from photoalbums.lib.ai_people_preprocess import _rembg_providers


def test_matcher_refreshes_legacy_reviewed_face_to_current_model(tmp_path, monkeypatch):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    person = store.add_person(name="Alice Example")
    image_path = tmp_path / "page.jpg"
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (70, 70), (220, 220, 220), -1)
    cv2.imwrite(str(image_path), image)

    face = store.add_face(
        embedding=[0.0, 1.0, 0.0],
        person_id=person["person_id"],
        source_type="photo",
        source_path=str(image_path),
        bbox=[10, 10, 40, 40],
        metadata={
            "embedding_model": "legacy.arcface",
            "detector_model": "legacy.detector",
        },
    )
    store.assign_face(
        face["face_id"],
        person["person_id"],
        reviewed_by_human=True,
        review_status="confirmed",
    )

    matcher = CastPeopleMatcher(cast_store_dir=store.root_dir, max_faces=1)
    monkeypatch.setattr(matcher, "_detect_faces", lambda image_bgr: [(10, 10, 40, 40)])
    monkeypatch.setattr(matcher, "_is_valid_face_crop", lambda crop_bgr: True)
    monkeypatch.setattr(matcher, "_arcface_embed", lambda crop_bgr: [1.0, 0.0, 0.0])
    monkeypatch.setattr(matcher, "_embed", lambda crop_bgr: [0.2, 0.8, 0.0])
    monkeypatch.setattr(matcher, "_estimate_quality", lambda crop_bgr: 0.93)

    matches = matcher.match_image(image_path, source_path=image_path)

    assert len(matches) == 1
    assert matches[0].name == "Alice Example"
    assert matches[0].certainty == 1.0
    assert matches[0].reviewed_by_human is True

    refreshed = store.get_face(face["face_id"])
    assert refreshed is not None
    assert refreshed["metadata"]["embedding_model"] == matcher._current_embedding_model
    assert refreshed["review_status"] == "confirmed"


def test_match_image_bbox_uses_original_image_coords_not_rescaled(tmp_path, monkeypatch):
    """Bounding box in PersonMatch and the face store must be in original-image pixel space.

    When a large image is processed, a rescaled copy may be used for AI caption/OCR models.
    Face detection always runs on the original image, so the stored bbox must correspond to
    the original image dimensions — not those of any rescaled copy.

    The mock detector returns different coordinates depending on which image it receives:
    - original (400x300): face at [100, 75, 80, 80]
    - rescaled  (200x150): face at [ 50, 37, 40, 40]

    A regression where match_image() received the rescaled path instead of the original
    would produce the wrong bbox and this test would catch it.
    """
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    # Original image: 400x300 with a bright rectangle representing the face region
    original_path = tmp_path / "page.jpg"
    original = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(original, (100, 75), (180, 155), (220, 220, 220), -1)
    cv2.imwrite(str(original_path), original)

    # Rescaled copy at exactly half size — face would appear at half the coordinates
    rescaled_path = tmp_path / "page_rescaled.jpg"
    rescaled = cv2.resize(original, (200, 150))
    cv2.imwrite(str(rescaled_path), rescaled)

    # Add a confirmed person so PersonMatch is populated (confirmed faces bypass the
    # suggestion pipeline and directly produce a PersonMatch with bbox).
    person = store.add_person(name="Bob Example")
    confirmed_face = store.add_face(
        embedding=[1.0, 0.0, 0.0],
        person_id=person["person_id"],
        source_type="photo",
        source_path=str(original_path),
        bbox=[100, 75, 80, 80],
        metadata={"embedding_model": "insightface.buffalo_l.arcface_512"},
    )
    store.assign_face(
        confirmed_face["face_id"],
        person["person_id"],
        reviewed_by_human=True,
        review_status="confirmed",
    )

    matcher = CastPeopleMatcher(cast_store_dir=store.root_dir, max_faces=5)

    def _size_aware_detect(image_bgr):
        h, w = image_bgr.shape[:2]
        if w == 400 and h == 300:
            return [(100, 75, 80, 80)]  # correct: original image
        if w == 200 and h == 150:
            return [(50, 37, 40, 40)]  # wrong: rescaled image
        return []

    monkeypatch.setattr(matcher, "_detect_faces", _size_aware_detect)
    monkeypatch.setattr(matcher, "_is_valid_face_crop", lambda crop_bgr: True)
    monkeypatch.setattr(matcher, "_arcface_embed", lambda crop_bgr: [1.0, 0.0, 0.0])
    monkeypatch.setattr(matcher, "_embed", lambda crop_bgr: [1.0, 0.0, 0.0])
    monkeypatch.setattr(matcher, "_estimate_quality", lambda crop_bgr: 0.85)

    matches = matcher.match_image(original_path, source_path=original_path)

    assert len(matches) == 1, f"Expected one match, got {matches}"
    assert matches[0].name == "Bob Example"
    assert matches[0].bbox == [100, 75, 80, 80], (
        f"PersonMatch.bbox should be in original (400x300) coordinates, "
        f"got {matches[0].bbox} — likely face detection ran on a rescaled image"
    )

    face_record = store.get_face(confirmed_face["face_id"])
    assert face_record is not None
    assert face_record["bbox"] == [100, 75, 80, 80], (
        f"Stored face bbox should be in original (400x300) coordinates, "
        f"got {face_record['bbox']} — likely face detection ran on a rescaled image"
    )


def test_new_face_is_added_to_cast_store(tmp_path, monkeypatch):
    """A face not yet in the Cast store should be added via store.add_face().

    This covers the path where match_image() detects a face that has no existing
    record, creates one, and queues it for review — so the face shows up in Cast.
    """
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    image_path = tmp_path / "page.jpg"
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (70, 70), (220, 220, 220), -1)
    cv2.imwrite(str(image_path), image)

    matcher = CastPeopleMatcher(cast_store_dir=store.root_dir, max_faces=5)
    monkeypatch.setattr(matcher, "_detect_faces", lambda image_bgr: [(10, 10, 60, 60)])
    monkeypatch.setattr(matcher, "_is_valid_face_crop", lambda crop_bgr: True)
    monkeypatch.setattr(matcher, "_arcface_embed", lambda crop_bgr: [1.0, 0.0, 0.0])
    monkeypatch.setattr(matcher, "_embed", lambda crop_bgr: [1.0, 0.0, 0.0])
    monkeypatch.setattr(matcher, "_estimate_quality", lambda crop_bgr: 0.80)
    monkeypatch.setattr(matcher, "_save_crop", lambda face_id, crop_bgr: f"crops/{face_id}.jpg")

    assert store.list_faces() == [], "store should be empty before match_image"

    matcher.match_image(image_path, source_path=image_path)

    faces = store.list_faces()
    assert len(faces) == 1, f"Expected 1 face added to Cast store, got {len(faces)}"
    assert faces[0]["source_type"] == "photo"
    assert faces[0]["source_path"] == str(image_path)
    assert faces[0]["bbox"] == [10, 10, 60, 60]


def test_match_image_recovery_refreshes_active_face_without_duplicate_row(tmp_path, monkeypatch):
    store = TextFaceStore(tmp_path / "cast_data")
    store.ensure_files()

    person = store.add_person(name="Leslie Cordell")
    reference_path = tmp_path / "reference.jpg"
    reference = np.full((120, 120, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(reference_path), reference)
    prototype_face = store.add_face(
        embedding=[1.0, 0.0, 0.0],
        person_id=person["person_id"],
        source_type="photo",
        source_path=str(reference_path),
        bbox=[10, 10, 60, 60],
        metadata={"embedding_model": "insightface.buffalo_l.arcface_512"},
    )
    store.assign_face(
        prototype_face["face_id"],
        person["person_id"],
        reviewed_by_human=True,
        review_status="confirmed",
    )

    image_path = tmp_path / "page.jpg"
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (70, 70), (30, 30, 30), -1)
    cv2.imwrite(str(image_path), image)

    matcher = CastPeopleMatcher(
        cast_store_dir=store.root_dir,
        max_faces=5,
        min_sample_count=1,
    )

    monkeypatch.setattr(matcher, "_detect_faces", lambda image_bgr: [(10, 10, 60, 60)])
    monkeypatch.setattr(matcher, "_is_valid_face_crop", lambda crop_bgr: True)
    monkeypatch.setattr(matcher, "_estimate_quality", lambda crop_bgr: 0.85)
    monkeypatch.setattr(matcher, "_save_crop", lambda face_id, crop_bgr: f"crops/{face_id}.jpg")
    monkeypatch.setattr(
        "photoalbums.lib.ai_people.build_rembg_bgr",
        lambda image_bgr: np.full_like(image_bgr, 255),
    )

    def _embedding_by_brightness(crop_bgr):
        if float(np.mean(crop_bgr)) > 200.0:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    monkeypatch.setattr(matcher, "_arcface_embed", _embedding_by_brightness)
    monkeypatch.setattr(matcher, "_embed", lambda crop_bgr: [0.0, 1.0, 0.0])

    matches = matcher.match_image(image_path, source_path=image_path)
    assert matches == []
    first_pass_faces = store.list_faces_for_source(str(image_path))
    assert len(first_pass_faces) == 1
    assert first_pass_faces[0]["person_id"] is None
    assert first_pass_faces[0]["metadata"]["analysis_variant"] == "original"

    recovery_matches = matcher.match_image_recovery(image_path, source_path=image_path)
    assert len(recovery_matches) == 1
    assert recovery_matches[0].name == "Leslie Cordell"

    recovered_faces = store.list_faces_for_source(str(image_path))
    assert len(recovered_faces) == 1
    assert recovered_faces[0]["metadata"]["analysis_variant"] == "rembg"


def test_rembg_providers_prefers_directml_when_available(monkeypatch):
    class _FakeOrt:
        @staticmethod
        def get_available_providers():
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(
        "photoalbums.lib.ai_people_preprocess._load_onnxruntime",
        lambda: _FakeOrt(),
    )

    assert _rembg_providers() == ["DmlExecutionProvider", "CPUExecutionProvider"]
