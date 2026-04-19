import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cli


class TestPhotoalbumsCLI(unittest.TestCase):
    def test_ai_forwards_remainder_without_index_subcommand(self):
        fake = mock.Mock()
        fake.run_ai_index.return_value = 0

        with mock.patch("cli._import_commands", return_value=fake):
            rc = cli.main(["ai", "--max-images", "3", "--dry-run"])

        self.assertEqual(rc, 0)
        fake.run_ai_index.assert_called_once_with(["--max-images", "3", "--dry-run"])

    def test_ai_index_forwards_remainder(self):
        fake = mock.Mock()
        fake.run_ai_index.return_value = 0

        with mock.patch("cli._import_commands", return_value=fake):
            rc = cli.main(["ai", "index", "--", "--max-images", "5", "--dry-run"])

        self.assertEqual(rc, 0)
        fake.run_ai_index.assert_called_once_with(["--max-images", "5", "--dry-run"])

    def test_ai_gps_forwards_reprocess_mode(self):
        fake = mock.Mock()
        fake.run_ai_index.return_value = 0

        with mock.patch("cli._import_commands", return_value=fake):
            rc = cli.main(["ai", "gps", "--max-images", "2", "--dry-run"])

        self.assertEqual(rc, 0)
        fake.run_ai_index.assert_called_once_with(["--reprocess-mode", "gps", "--max-images", "2", "--dry-run"])

    def test_ai_help_forwards_to_ai_index(self):
        fake = mock.Mock()
        fake.run_ai_index.return_value = 0

        with mock.patch("cli._import_commands", return_value=fake):
            rc = cli.main(["ai", "--help"])

        self.assertEqual(rc, 0)
        fake.run_ai_index.assert_called_once_with(["--help"])

    def test_ai_steps_prints_pipeline(self):
        fake = mock.Mock()
        fake.run_ai_index.return_value = 0

        with mock.patch("cli._import_commands", return_value=fake), mock.patch("builtins.print") as print_mock:
            rc = cli.main(["ai", "steps"])

        self.assertEqual(rc, 0)
        fake.run_ai_index.assert_not_called()
        print_mock.assert_any_call("AI pipeline steps:")

    def test_metadata_apply_dispatch(self):
        fake = mock.Mock()
        fake.run_apply_metadata.return_value = 0

        with mock.patch("cli._import_commands", return_value=fake):
            rc = cli.main(["metadata", "apply"])

        self.assertEqual(rc, 0)
        fake.run_apply_metadata.assert_called_once_with()

    def test_checksum_tree_dispatch(self):
        fake = mock.Mock()
        fake.run_checksum_tree.return_value = 0

        with mock.patch("cli._import_commands", return_value=fake):
            rc = cli.main(["checksum", "tree", "Photo Albums", "--verify"])

        self.assertEqual(rc, 0)
        fake.run_checksum_tree.assert_called_once_with(base_dir="Photo Albums", verify=True)

    def test_face_refresh_dispatch(self):
        fake = mock.Mock()
        fake.run_face_refresh.return_value = 0

        with mock.patch("cli._import_commands", return_value=fake):
            rc = cli.main(["face-refresh", "Egypt_1975", "--photos-root", "Photo Albums", "--page", "9", "--force"])

        self.assertEqual(rc, 0)
        fake.run_face_refresh.assert_called_once_with(
            album_id="Egypt_1975",
            photos_root="Photo Albums",
            page="9",
            force=True,
        )

    def test_migrate_caption_layout_dispatch(self):
        fake = mock.Mock()
        fake.run_migrate_caption_layout.return_value = 0

        with mock.patch("cli._import_commands", return_value=fake):
            rc = cli.main(["migrate-caption-layout", "--photos-root", "Photo Albums", "--verify-only"])

        self.assertEqual(rc, 0)
        fake.run_migrate_caption_layout.assert_called_once_with(
            photos_root="Photo Albums",
            verify_only=True,
        )

    def test_repair_crop_source_dispatch(self):
        fake = mock.Mock()
        fake.run_repair_crop_source.return_value = 0

        with mock.patch("cli._import_commands", return_value=fake):
            rc = cli.main(["repair-crop-source", "Portugal_1988_B00", "--photos-root", "Photo Albums", "--page", "23"])

        self.assertEqual(rc, 0)
        fake.run_repair_crop_source.assert_called_once_with(
            album_id="Portugal_1988_B00",
            photos_root="Photo Albums",
            page="23",
            verify_only=False,
        )

    def test_repair_crop_numbers_dispatch(self):
        fake = mock.Mock()
        fake.run_repair_crop_numbers.return_value = 0

        with mock.patch("cli._import_commands", return_value=fake):
            rc = cli.main(["repair-crop-numbers", "Family_1907-1946_B01", "--photos-root", "Photo Albums", "--page", "40"])

        self.assertEqual(rc, 0)
        fake.run_repair_crop_numbers.assert_called_once_with(
            album_id="Family_1907-1946_B01",
            photos_root="Photo Albums",
            page="40",
        )

    def test_repair_page_derived_views_dispatch(self):
        fake = mock.Mock()
        fake.run_repair_page_derived_views.return_value = 0

        with mock.patch("cli._import_commands", return_value=fake):
            rc = cli.main(
                ["repair-page-derived-views", "Family_1907-1946_B01", "--photos-root", "Photo Albums", "--page", "40"]
            )

        self.assertEqual(rc, 0)
        fake.run_repair_page_derived_views.assert_called_once_with(
            album_id="Family_1907-1946_B01",
            photos_root="Photo Albums",
            page="40",
        )


if __name__ == "__main__":
    unittest.main()
