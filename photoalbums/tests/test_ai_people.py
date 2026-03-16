from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cast.ingest import CURRENT_FACE_EMBEDDING_MODEL
from cast.storage import TextFaceStore
from photoalbums.lib.ai_people import CastPeopleMatcher


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
    monkeypatch.setattr(
        matcher._ingestor,
        "is_valid_face_crop",
        lambda crop_bgr, skip_artwork=True: True,
    )
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
    assert refreshed["metadata"]["embedding_model"] == CURRENT_FACE_EMBEDDING_MODEL
    assert refreshed["review_status"] == "confirmed"


def test_match_image_bbox_uses_original_image_coords_not_rescaled(
    tmp_path, monkeypatch
):
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
    monkeypatch.setattr(
        matcher._ingestor,
        "is_valid_face_crop",
        lambda crop_bgr, skip_artwork=True: True,
    )
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
