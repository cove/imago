"""Tests for _update_region_captions_from_metadata in ai_index_analysis.

Covers the regression where mwg-rs:Name and imago:PhotoNumber were never written
to the page-view's regions XMP because the function was being passed the temp
content_path instead of the original page-view path.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from photoalbums.lib.ai_index_analysis import _update_region_captions_from_metadata
from photoalbums.lib.ai_view_regions import RegionResult, RegionWithCaption
from photoalbums.lib.xmp_sidecar import read_region_list, write_region_list


def _make_minimal_jpeg(path: Path, width: int = 800, height: int = 600) -> None:
    from PIL import Image

    Image.new("RGB", (width, height), color=(200, 200, 200)).save(str(path), format="JPEG", quality=85)


def _seed_regions_xmp(image_path: Path, count: int) -> Path:
    """Write an XMP sidecar with `count` empty-named regions next to image_path."""
    xmp_path = image_path.with_suffix(".xmp")
    img_w, img_h = 800, 600
    rwcs = [
        RegionWithCaption(
            RegionResult(
                index=i,
                x=i * 100,
                y=0,
                width=100,
                height=100,
                caption_hint="",
                person_names=[],
                photo_number=0,
            ),
            caption="",
        )
        for i in range(count)
    ]
    write_region_list(xmp_path, rwcs, img_w, img_h)
    return xmp_path


class TestUpdateRegionCaptionsFromMetadata(unittest.TestCase):
    def test_writes_captions_and_photo_numbers_to_xmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page.jpg"
            _make_minimal_jpeg(image)
            xmp_path = _seed_regions_xmp(image, count=2)

            # Simulate the metadata step output's photo_captions list.
            photo_captions = [
                {"photo_number": 1, "caption": "Smiling on the beach"},
                {"photo_number": 2, "caption": "By the lighthouse"},
            ]
            _update_region_captions_from_metadata(image, photo_captions)

            regions = read_region_list(xmp_path, 800, 600)
            self.assertEqual(len(regions), 2)
            self.assertEqual(regions[0]["caption"], "Smiling on the beach")
            self.assertEqual(regions[0]["photo_number"], 1)
            self.assertEqual(regions[1]["caption"], "By the lighthouse")
            self.assertEqual(regions[1]["photo_number"], 2)

    def test_does_nothing_when_xmp_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "no-sidecar.jpg"
            _make_minimal_jpeg(image)
            # No XMP written next to image
            _update_region_captions_from_metadata(
                image, [{"photo_number": 1, "caption": "ignored"}]
            )
            self.assertFalse(image.with_suffix(".xmp").exists())

    def test_does_nothing_when_photo_captions_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page.jpg"
            _make_minimal_jpeg(image)
            xmp_path = _seed_regions_xmp(image, count=1)
            mtime_before = xmp_path.stat().st_mtime_ns

            _update_region_captions_from_metadata(image, [])

            self.assertEqual(xmp_path.stat().st_mtime_ns, mtime_before)

    def test_assigns_photo_number_even_when_caption_is_empty(self):
        """AI may return a valid photo_number with an empty caption (no typed
        text on the page); the photo_number should still propagate."""
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page.jpg"
            _make_minimal_jpeg(image)
            xmp_path = _seed_regions_xmp(image, count=2)

            _update_region_captions_from_metadata(
                image,
                [
                    {"photo_number": 1, "caption": ""},
                    {"photo_number": 2, "caption": ""},
                ],
            )

            regions = read_region_list(xmp_path, 800, 600)
            self.assertEqual(regions[0]["photo_number"], 1)
            self.assertEqual(regions[1]["photo_number"], 2)
            self.assertEqual(regions[0]["caption"], "")
            self.assertEqual(regions[1]["caption"], "")

    def test_ai_caption_overwrites_existing_hint(self):
        """The metadata AI's numbered caption is the authoritative region name."""
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page.jpg"
            _make_minimal_jpeg(image)
            xmp_path = image.with_suffix(".xmp")
            existing = RegionWithCaption(
                RegionResult(
                    index=0,
                    x=0,
                    y=0,
                    width=100,
                    height=100,
                    caption_hint="Manually set caption",
                ),
                caption="Manually set caption",
            )
            write_region_list(xmp_path, [existing], 800, 600)

            _update_region_captions_from_metadata(
                image, [{"photo_number": 1, "caption": "AI override"}]
            )

            regions = read_region_list(xmp_path, 800, 600)
            self.assertEqual(regions[0]["caption"], "AI override")
            self.assertEqual(regions[0]["caption_hint"], "AI override")
            self.assertEqual(regions[0]["photo_number"], 1)

    def test_does_not_fall_back_to_caption_hint_when_ai_caption_empty(self):
        """When the AI returns an empty caption, mwg-rs:Name stays empty —
        we don't silently substitute caption_hint or any other field."""
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page.jpg"
            _make_minimal_jpeg(image)
            xmp_path = image.with_suffix(".xmp")
            # Region has a non-empty caption_hint but AI returns empty caption.
            seeded = RegionWithCaption(
                RegionResult(
                    index=0,
                    x=0,
                    y=0,
                    width=100,
                    height=100,
                    caption_hint="Stored hint should not leak into Name",
                    photo_number=1,
                ),
                caption="",
            )
            write_region_list(xmp_path, [seeded], 800, 600)

            _update_region_captions_from_metadata(
                image, [{"photo_number": 1, "caption": ""}]
            )

            regions = read_region_list(xmp_path, 800, 600)
            self.assertEqual(regions[0]["caption"], "")
            self.assertEqual(regions[0]["photo_number"], 1)

    def test_persists_person_names_through_update(self):
        """The bag-shadowing bug in write_region_list previously corrupted
        the XML structure when a region had person_names. Verify that
        the names round-trip correctly across an update."""
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page.jpg"
            _make_minimal_jpeg(image)
            xmp_path = image.with_suffix(".xmp")
            seeded = [
                RegionWithCaption(
                    RegionResult(
                        index=0, x=0, y=0, width=100, height=100,
                        person_names=["Alice", "Bob"],
                    ),
                    caption="",
                ),
                RegionWithCaption(
                    RegionResult(
                        index=1, x=200, y=0, width=100, height=100,
                        person_names=["Carol"],
                    ),
                    caption="",
                ),
            ]
            write_region_list(xmp_path, seeded, 800, 600)

            _update_region_captions_from_metadata(
                image,
                [
                    {"photo_number": 1, "caption": "First"},
                    {"photo_number": 2, "caption": "Second"},
                ],
            )

            regions = read_region_list(xmp_path, 800, 600)
            self.assertEqual(len(regions), 2)
            self.assertEqual(regions[0]["caption"], "First")
            self.assertEqual(regions[1]["caption"], "Second")
            self.assertEqual(sorted(regions[0]["person_names"]), ["Alice", "Bob"])
            self.assertEqual(regions[1]["person_names"], ["Carol"])


class TestMetadataInputHashVersioning(unittest.TestCase):
    SETTINGS = {
        "cast_store_signature": "",
        "caption_engine": "lmstudio",
        "caption_model": "test-model",
        "nominatim_base_url": "",
        "model": "",
        "enable_objects": False,
        "crop_paths_signature": "",
    }

    def test_input_hash_includes_version_tag(self):
        """The metadata input hash carries a version tag so caches written
        before the regions-write fix are invalidated automatically. Without
        this, users would otherwise need to manually force-re-run the step."""
        from photoalbums.lib import ai_index_steps as steps_mod

        # The outer _sha16 call combines a version tag, model, base url, and
        # the prompt-asset hashes. The version tag is the load-bearing bit
        # that invalidates pre-fix caches; capture the outer call and assert
        # its first arg starts with "v".
        original = steps_mod._sha16
        outer_calls: list[tuple] = []

        def _spy(*args):
            # Outer metadata_input_hash call passes >= 3 args: version,
            # model, base_url, prompt_payload. The inner _prompt_hash_payload
            # call passes a single (already-serialized) arg.
            if len(args) > 1:
                outer_calls.append(args)
            return original(*args)

        with mock.patch.object(steps_mod, "_sha16", side_effect=_spy):
            steps_mod.metadata_input_hash(self.SETTINGS, {})
        self.assertTrue(outer_calls, "expected an outer _sha16 invocation")
        first_arg = outer_calls[0][0]
        self.assertTrue(
            isinstance(first_arg, str) and first_arg.startswith("v"),
            f"metadata_input_hash should lead with a version tag; got {first_arg!r}",
        )

    def test_input_hash_is_deterministic(self):
        from photoalbums.lib.ai_index_steps import metadata_input_hash

        self.assertEqual(
            metadata_input_hash(self.SETTINGS, {}),
            metadata_input_hash(self.SETTINGS, {}),
        )


class TestWriteRegionListAppliesStoredAIPhotos(unittest.TestCase):
    """When detect-regions (or any caller) writes regions with empty
    captions, write_region_list should look up detections.caption.photos
    and populate mwg-rs:Name from the AI's stored response, keyed by
    photo_number."""

    def _seed_xmp_with_detections(self, image: Path, photos: list[dict]) -> Path:
        """Write an XMP with imago:Detections containing the given AI photos."""
        from photoalbums.lib.xmp_sidecar import write_xmp_sidecar

        xmp_path = image.with_suffix(".xmp")
        write_xmp_sidecar(
            xmp_path,
            person_names=[],
            subjects=[],
            description="",
            ocr_text="",
            detections_payload={
                "caption": {
                    "effective_engine": "lmstudio",
                    "model": "test-model",
                    "photos": photos,
                }
            },
        )
        return xmp_path

    def test_write_region_list_uses_stored_ai_caption_when_caption_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page.jpg"
            _make_minimal_jpeg(image)
            xmp_path = self._seed_xmp_with_detections(
                image,
                [
                    {"photo_number": 1, "caption": "Oxford Street, London"},
                    {"photo_number": 2, "caption": "Regent Street, London"},
                ],
            )

            # Caller passes empty captions but sets photo_number per region —
            # this is the post-detect-regions state.
            rwcs = [
                RegionWithCaption(
                    RegionResult(
                        index=0, x=0, y=0, width=100, height=100, photo_number=1,
                    ),
                    caption="",
                ),
                RegionWithCaption(
                    RegionResult(
                        index=1, x=200, y=0, width=100, height=100, photo_number=2,
                    ),
                    caption="",
                ),
            ]
            write_region_list(xmp_path, rwcs, 800, 600)

            regions = read_region_list(xmp_path, 800, 600)
            self.assertEqual(regions[0]["caption"], "Oxford Street, London")
            self.assertEqual(regions[0]["photo_number"], 1)
            self.assertEqual(regions[1]["caption"], "Regent Street, London")
            self.assertEqual(regions[1]["photo_number"], 2)

    def test_explicit_caption_takes_precedence_over_stored(self):
        """If the caller explicitly sets RegionWithCaption.caption, it wins
        over whatever's in detections.caption.photos."""
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page.jpg"
            _make_minimal_jpeg(image)
            xmp_path = self._seed_xmp_with_detections(
                image, [{"photo_number": 1, "caption": "Stored AI caption"}]
            )

            rwcs = [
                RegionWithCaption(
                    RegionResult(
                        index=0, x=0, y=0, width=100, height=100, photo_number=1,
                    ),
                    caption="Caller override",
                )
            ]
            write_region_list(xmp_path, rwcs, 800, 600)

            regions = read_region_list(xmp_path, 800, 600)
            self.assertEqual(regions[0]["caption"], "Caller override")

    def test_no_stored_photos_leaves_name_empty(self):
        """No detections.caption.photos and no explicit caption → empty Name."""
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page.jpg"
            _make_minimal_jpeg(image)
            xmp_path = image.with_suffix(".xmp")

            rwcs = [
                RegionWithCaption(
                    RegionResult(
                        index=0, x=0, y=0, width=100, height=100, photo_number=1,
                    ),
                    caption="",
                )
            ]
            write_region_list(xmp_path, rwcs, 800, 600)

            regions = read_region_list(xmp_path, 800, 600)
            self.assertEqual(regions[0]["caption"], "")
            self.assertEqual(regions[0]["photo_number"], 1)

    def test_empty_stored_caption_does_not_overwrite_explicit(self):
        """An empty stored caption should not nullify a non-empty caller value."""
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page.jpg"
            _make_minimal_jpeg(image)
            xmp_path = self._seed_xmp_with_detections(
                image, [{"photo_number": 1, "caption": ""}]
            )

            rwcs = [
                RegionWithCaption(
                    RegionResult(
                        index=0, x=0, y=0, width=100, height=100, photo_number=1,
                    ),
                    caption="From caller",
                )
            ]
            write_region_list(xmp_path, rwcs, 800, 600)

            regions = read_region_list(xmp_path, 800, 600)
            self.assertEqual(regions[0]["caption"], "From caller")


class TestMetadataResponseParsing(unittest.TestCase):
    """The metadata response parser must tolerate <think>...</think> reasoning
    blocks that some chat templates emit inline in message.content even with
    structured outputs requested."""

    def test_strips_paired_think_block_before_json(self):
        from photoalbums.lib.ai_metadata import _parse_metadata_response

        raw = (
            "<think>The user wants per-photo metadata. I see 2 numbered "
            "boxes. Photo 1 is the beach scene...</think>\n"
            '{"photos": [{"photo_number": 1, "location": "Spain", '
            '"location_name": "Barcelona", "est_date": "1988", '
            '"scene_ocr": "", "caption": "On the beach", "people_count": 2}]}'
        )
        result = _parse_metadata_response(raw)
        self.assertEqual(len(result.photos), 1)
        self.assertEqual(result.photos[0].photo_number, 1)
        self.assertEqual(result.photos[0].caption, "On the beach")

    def test_strips_dangling_think_open_with_no_closer(self):
        from photoalbums.lib.ai_metadata import _parse_metadata_response

        # Some templates open <think> but the closer falls off — we should
        # still find the JSON that follows.
        raw = (
            "<think>Reasoning that never closes properly because the model "
            "ran out of budget partway through ...\n"
            '{"photos": [{"photo_number": 2, "location": "", '
            '"location_name": "", "est_date": "", "scene_ocr": "", '
            '"caption": "Lighthouse", "people_count": 0}]}'
        )
        result = _parse_metadata_response(raw)
        self.assertEqual(len(result.photos), 1)
        self.assertEqual(result.photos[0].caption, "Lighthouse")

    def test_falls_back_to_extraction_when_outer_json_is_wrong_schema(self):
        from photoalbums.lib.ai_metadata import _parse_metadata_response

        # Top-level parse succeeds on a reasoning preamble that happens to be
        # valid JSON — we should still find the real metadata payload.
        raw = (
            '{"thoughts": "I will now produce the answer"} '
            '{"photos": [{"photo_number": 3, "location": "", '
            '"location_name": "", "est_date": "", "scene_ocr": "", '
            '"caption": "Castle", "people_count": 1}]}'
        )
        result = _parse_metadata_response(raw)
        self.assertEqual(len(result.photos), 1)
        self.assertEqual(result.photos[0].caption, "Castle")
        self.assertEqual(result.photos[0].photo_number, 3)

    def test_clean_json_passes_through_unchanged(self):
        from photoalbums.lib.ai_metadata import _parse_metadata_response

        raw = (
            '{"photos": [{"photo_number": 1, "location": "", '
            '"location_name": "", "est_date": "", "scene_ocr": "", '
            '"caption": "Hello", "people_count": 0}]}'
        )
        result = _parse_metadata_response(raw)
        self.assertEqual(result.photos[0].caption, "Hello")

    def test_request_payload_disables_thinking(self):
        """The metadata engine sends chat_template_kwargs.enable_thinking=false
        so thinking-capable templates skip emitting reasoning."""
        from photoalbums.lib import ai_metadata

        captured: dict = {}

        def _fake_request(_url, *, payload, timeout):  # noqa: ARG001
            captured["payload"] = payload
            return {
                "choices": [
                    {
                        "message": {"content": '{"photos": []}'},
                        "finish_reason": "stop",
                    }
                ]
            }

        engine = ai_metadata.MetadataEngine(model_name="test-model")
        engine.base_url = "http://localhost:1234/v1"

        with mock.patch.object(ai_metadata, "_lmstudio_request_json", side_effect=_fake_request):
            with mock.patch.object(
                ai_metadata, "_build_data_url", return_value="data:image/jpeg;base64,xx"
            ):
                engine.analyze(Path("/dev/null"))

        self.assertIn("chat_template_kwargs", captured["payload"])
        self.assertEqual(
            captured["payload"]["chat_template_kwargs"], {"enable_thinking": False}
        )


if __name__ == "__main__":
    unittest.main()
