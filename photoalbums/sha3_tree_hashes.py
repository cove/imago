import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

MANIFEST_NAME = "SHA256SUMS"
TOP_MANIFEST_NAME = "SHA256SUMS"

def iter_files(base_dir: Path) -> Iterable[Path]:
    for root, _, files in os.walk(base_dir):
        for name in files:
            yield Path(root) / name


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_manifest(manifest_path: Path, entries: List[Tuple[str, Path]]) -> None:
    lines = []
    for digest, rel_path in entries:
        # BSD-style format, compatible with tools like rhash
        lines.append(f"SHA256 ({rel_path.as_posix()}) = {digest}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def parse_manifest(manifest_path: Path) -> List[Tuple[str, Path]]:
    entries: List[Tuple[str, Path]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("SHA256 (") or ") = " not in line:
            raise ValueError(f"Invalid manifest line in {manifest_path}: {line}")
        prefix = "SHA256 ("
        path_part, digest = line[len(prefix):].split(") = ", 1)
        entries.append((digest.strip(), Path(path_part)))
    return entries


def album_dirs(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.iterdir() if p.is_dir()])


def should_skip_file(file_path: Path) -> bool:
    return (
        file_path.name in {MANIFEST_NAME, TOP_MANIFEST_NAME, ".DS_Store"}
        or file_path.is_symlink()
    )


def build_album_entries(album_dir: Path) -> List[Tuple[str, Path]]:
    entries: List[Tuple[str, Path]] = []

    for file_path in sorted(iter_files(album_dir)):
        if should_skip_file(file_path):
            continue
        rel_path = file_path.relative_to(album_dir)
        digest = sha256_file(file_path)
        entries.append((digest, rel_path))

    return entries


def build_album_manifest(album_dir: Path, entries: List[Tuple[str, Path]]) -> Path:
    manifest_path = album_dir / MANIFEST_NAME
    write_manifest(manifest_path, entries)
    return manifest_path


def build_top_manifest(base_dir: Path, entries: List[Tuple[str, Path]]) -> Path:
    top_manifest_path = base_dir / TOP_MANIFEST_NAME
    write_manifest(top_manifest_path, entries)
    return top_manifest_path


def check_manifest(manifest_path: Path, digest_cache: Dict[Path, str]) -> List[str]:
    errors: List[str] = []
    for expected_digest, rel_path in parse_manifest(manifest_path):
        target = (manifest_path.parent / rel_path).resolve()
        if not target.exists():
            errors.append(f"Missing file: {target}")
            continue
        actual_digest = digest_cache.get(target)
        if actual_digest is None:
            actual_digest = sha256_file(target)
            digest_cache[target] = actual_digest
        if actual_digest.lower() != expected_digest.lower():
            errors.append(f"Hash mismatch: {target}")
    return errors


def verify_tree(base_dir: Path) -> int:
    failures: List[str] = []
    digest_cache: Dict[Path, str] = {}
    for album_dir in album_dirs(base_dir):
        manifest_path = album_dir / MANIFEST_NAME
        if not manifest_path.exists():
            failures.append(f"Missing manifest: {manifest_path}")
            continue
        failures.extend(check_manifest(manifest_path, digest_cache))

    top_manifest_path = base_dir / TOP_MANIFEST_NAME
    if not top_manifest_path.exists():
        failures.append(f"Missing manifest: {top_manifest_path}")
    else:
        failures.extend(check_manifest(top_manifest_path, digest_cache))

    if failures:
        print("FAILED")
        for msg in failures:
            print(f"- {msg}")
        return 1

    print("OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate hierarchical SHA-256 manifests for a Photo Albums directory tree.",
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=".",
        help="Path to the Photo Albums directory (default: current directory).",
    )
    parser.add_argument(
        "--verify",
        "--check",
        dest="verify",
        action="store_true",
        help="Verify hashes against existing manifests instead of generating them.",
    )

    args = parser.parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()

    if not base_dir.is_dir():
        raise SystemExit(f"Not a directory: {base_dir}")

    if args.verify:
        return verify_tree(base_dir)

    top_entries: List[Tuple[str, Path]] = []
    for album_dir in album_dirs(base_dir):
        album_entries = build_album_entries(album_dir)
        build_album_manifest(album_dir, album_entries)
        top_entries.extend(
            (digest, album_dir.relative_to(base_dir) / rel_path)
            for digest, rel_path in album_entries
        )

    build_top_manifest(base_dir, sorted(top_entries, key=lambda item: item[1].as_posix()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
