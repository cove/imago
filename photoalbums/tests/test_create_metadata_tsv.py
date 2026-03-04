import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import create_metadata_tsv


class TestCreateMetadataTSV(unittest.TestCase):
    def test_find_files_filters_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Album_Archive"
            view = base / "Album_View"
            archive.mkdir()
            view.mkdir()
            (archive / "a.tif").touch()
            (archive / "b.jpg").touch()
            (archive / "c.txt").touch()
            (view / "d.tif").touch()

            files = create_metadata_tsv.find_files(base, create_metadata_tsv.FILE_EXTENSIONS)
            names = sorted(p.name for p in files)

            self.assertEqual(names, ["a.tif", "b.jpg"])

    def test_collect_all_metadata_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            file_path = base / "sub" / "a.tif"
            file_path.parent.mkdir()
            file_path.touch()

            with mock.patch("create_metadata_tsv.PHOTO_ALBUMS_DIR", base), mock.patch(
                "create_metadata_tsv.extract_metadata",
                return_value={"XMP-dc:Description": "desc"},
            ):
                results = create_metadata_tsv.collect_all_metadata([file_path], {}, {}, progress=False)

            self.assertEqual(len(results), 1)
            expected = str(Path("Photo Albums") / "sub" / "a.tif")
            self.assertEqual(results[0]["FilePath"], expected)

    def test_collect_all_metadata_overwrites_scan_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Family_1947-1953_B02_Archive"
            archive.mkdir()
            file_path = archive / "Family_1947-1953_B02_P33_S01.tif"
            file_path.touch()

            scan_totals = {archive: {33: 2}}

            with mock.patch("create_metadata_tsv.PHOTO_ALBUMS_DIR", base), mock.patch(
                "create_metadata_tsv.extract_metadata",
                return_value={
                    "XMP-dc:Description": "old",
                    "XMP-dc:Creator": "Someone Else",
                },
            ):
                results = create_metadata_tsv.collect_all_metadata(
                    [file_path],
                    scan_totals,
                    {},
                    progress=False,
                )

            self.assertEqual(len(results), 1)
            self.assertEqual(
                results[0]["XMP-dc:Description"],
                "Family (1947-1953) - Book 02, Page 33, Scan S01 of 2 total",
            )
            self.assertEqual(results[0]["XMP-dc:Creator"], create_metadata_tsv.CREATOR)
            self.assertEqual(results[0]["IPTC:Keywords"], ["Family"])

    def test_collect_all_metadata_non_matching_description_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Album_Archive"
            archive.mkdir()
            file_path = archive / "random_name.tif"
            file_path.touch()

            with mock.patch("create_metadata_tsv.PHOTO_ALBUMS_DIR", base), mock.patch(
                "create_metadata_tsv.extract_metadata",
                return_value={"XMP-dc:Description": "old"},
            ):
                results = create_metadata_tsv.collect_all_metadata([file_path], {}, {}, progress=False)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["XMP-dc:Description"], "")
            self.assertEqual(results[0]["XMP-dc:Creator"], create_metadata_tsv.CREATOR)

    def test_collect_all_metadata_overwrites_derived_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "Family_1947-1953_B02_Archive"
            archive.mkdir()
            file_path = archive / "Family_1947-1953_B02_P33_D01_02.tif"
            file_path.touch()

            derived_totals = {archive: {33: {"01": 3}}}

            with mock.patch("create_metadata_tsv.PHOTO_ALBUMS_DIR", base), mock.patch(
                "create_metadata_tsv.extract_metadata",
                return_value={
                    "XMP-dc:Description": "old",
                    "XMP-dc:Creator": "Someone Else",
                },
            ):
                results = create_metadata_tsv.collect_all_metadata(
                    [file_path],
                    {},
                    derived_totals,
                    progress=False,
                )

            self.assertEqual(len(results), 1)
            self.assertEqual(
                results[0]["XMP-dc:Description"],
                "Family (1947-1953) - Book 02, Page 33, Derived D01_02 of 3 total",
            )
            self.assertEqual(results[0]["XMP-dc:Creator"], create_metadata_tsv.CREATOR)
            self.assertEqual(results[0]["IPTC:Keywords"], ["Family"])

    def test_collect_all_metadata_adds_collection_keyword_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "SouthAmerica_1986_B00_Archive"
            archive.mkdir()
            file_path = archive / "SouthAmerica_1986_B00_P01_S01.tif"
            file_path.touch()

            scan_totals = {archive: {1: 1}}

            with mock.patch("create_metadata_tsv.PHOTO_ALBUMS_DIR", base), mock.patch(
                "create_metadata_tsv.extract_metadata",
                return_value={"IPTC:Keywords": ["Existing"]},
            ):
                results = create_metadata_tsv.collect_all_metadata(
                    [file_path],
                    scan_totals,
                    {},
                    progress=False,
                )

            self.assertEqual(results[0]["IPTC:Keywords"], ["Existing", "South America"])

    def test_collect_all_metadata_splits_multi_phrase_collection_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "EasternEuropeSpainMorocco_1986_B00_Archive"
            archive.mkdir()
            file_path = archive / "EasternEuropeSpainMorocco_1986_B00_P01_S01.tif"
            file_path.touch()

            scan_totals = {archive: {1: 1}}

            with mock.patch("create_metadata_tsv.PHOTO_ALBUMS_DIR", base), mock.patch(
                "create_metadata_tsv.extract_metadata",
                return_value={"IPTC:Keywords": ["Existing"]},
            ):
                results = create_metadata_tsv.collect_all_metadata(
                    [file_path],
                    scan_totals,
                    {},
                    progress=False,
                )

            self.assertEqual(
                results[0]["IPTC:Keywords"],
                ["Existing", "Eastern Europe", "Spain", "Morocco"],
            )

    def test_write_tsv_outputs_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "metadata.tsv"
            data = [{"FilePath": "Photo Albums/a.tif", "XMP-dc:Description": "desc"}]

            create_metadata_tsv.write_tsv(data, str(output_path))

            self.assertTrue(output_path.exists())
            content = output_path.read_text(encoding="utf-8")
            self.assertIn("FilePath", content.splitlines()[0])
            self.assertIn("Photo Albums/a.tif", content)


if __name__ == "__main__":
    unittest.main()
