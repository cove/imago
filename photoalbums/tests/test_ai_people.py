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
