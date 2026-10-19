"""Renumber all TIFs in a bennette _Archive dir so every scan is S01 (sequential pages)."""
import re
import sys
import uuid
from pathlib import Path

PATTERN = re.compile(r"^(?P<prefix>.+)_P(?P<page>\d+)_S(?P<scan>\d+)\.tif$", re.IGNORECASE)


def sort_key(f: Path):
    m = PATTERN.match(f.name)
    return (int(m.group("page")), int(m.group("scan")))


def run(archive_dir: Path, *, dry_run: bool = True) -> None:
    files = sorted(
        [f for f in archive_dir.iterdir() if f.suffix.lower() == ".tif" and PATTERN.match(f.name)],
        key=sort_key,
    )
    if not files:
        print("No matching TIF files found.")
        return

    prefix = PATTERN.match(files[0].name).group("prefix")
    renames = [(f, archive_dir / f"{prefix}_P{i:02d}_S01.tif") for i, f in enumerate(files, 1)]

    needs_change = [(old, new) for old, new in renames if old != new]
    if not needs_change:
        print("All files already correctly named.")
        return

    print(f"{'DRY RUN — ' if dry_run else ''}Renaming {len(needs_change)} of {len(files)} files:")
    for old, new in needs_change:
        print(f"  {old.name}  ->  {new.name}")

    if dry_run:
        print("\nRun with --apply to execute.")
        return

    tag = uuid.uuid4().hex[:8]
    tmp_names = [(old, archive_dir / f"_tmp_{tag}_{i}.tif") for i, (old, _) in enumerate(needs_change)]

    for old, tmp in tmp_names:
        old.rename(tmp)
    for (_, new), (_, tmp) in zip(needs_change, tmp_names):
        tmp.rename(new)

    print("Done.")


if __name__ == "__main__":
    args = sys.argv[1:]
    apply = "--apply" in args
    dirs = [Path(a) for a in args if not a.startswith("--")]
    if not dirs:
        dirs = [Path(r"C:\Users\covec\Videos\bennette_albums\FriendsInHighPlaces_Archive")]
    for d in dirs:
        print(f"\n{d}")
        run(d, dry_run=not apply)
