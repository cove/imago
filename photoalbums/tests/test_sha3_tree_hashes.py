import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sha3_tree_hashes as hashes


class TestSha3TreeHashes(unittest.TestCase):
    def test_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manifest = base / "SHA256SUMS"
            test_file = base / "a.txt"
            test_file.write_text("data", encoding="utf-8")

            digest = hashes.sha256_file(test_file)
            hashes.write_manifest(manifest, [(digest, Path("a.txt"))])

            parsed = hashes.parse_manifest(manifest)
            self.assertEqual(parsed, [(digest, Path("a.txt"))])

    def test_verify_tree_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            album = base / "Album1"
            album.mkdir()
            file_path = album / "a.txt"
            file_path.write_text("data", encoding="utf-8")

            entries = hashes.build_album_entries(album)
            hashes.build_album_manifest(album, entries)

            top_entries = [
                (digest, album.relative_to(base) / rel_path) for digest, rel_path in entries
            ]
            hashes.build_top_manifest(base, top_entries)

            self.assertEqual(hashes.verify_tree(base), 0)


if __name__ == "__main__":
    unittest.main()
