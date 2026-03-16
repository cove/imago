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

        with mock.patch("cli._import_commands", return_value=fake), mock.patch(
            "builtins.print"
        ) as print_mock:
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
        fake.run_checksum_tree.assert_called_once_with(
            base_dir="Photo Albums", verify=True
        )


if __name__ == "__main__":
    unittest.main()
