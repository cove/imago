from __future__ import annotations

from pathlib import Path

from vhs_pipeline.checksum import generate_drive_checksum, verify_archive, verify_drive
from vhs_pipeline.compare import run_comparisons
from vhs_pipeline.convert import (
    convert_avi_to_archive,
    convert_umatic_to_archive,
    embed_metadata_into_archives,
)
from vhs_pipeline.metadata import (
    convert_all_ffmetadata_to_chapters_tsv,
    generate_archive_metadata,
)
from vhs_pipeline.people_prefill import (
    apply_prefill_entries_to_people_tsv,
    prefill_people_from_cast,
    write_prefill_audit_tsv,
)
from vhs_pipeline.proxy import make_proxies
from vhs_pipeline.render import run_render, run_render_subtitles
from apps.plain_html_wizard.server import run as run_tuner_server


def run_convert_avi(paths):
    convert_avi_to_archive(paths)
    return 0


def run_convert_umatic(paths):
    convert_umatic_to_archive(paths)
    return 0


def run_embed_metadata(paths):
    embed_metadata_into_archives(paths)
    return 0


def run_generate_archive_metadata():
    return int(generate_archive_metadata() or 0)


def run_convert_ffmetadata_to_chapters_tsv(overwrite: bool = False):
    convert_all_ffmetadata_to_chapters_tsv(overwrite=bool(overwrite))
    return 0


def run_verify_archive(argv):
    return int(verify_archive(argv) or 0)


def run_make_proxies(show_frame_number: bool = False):
    return int(make_proxies(show_frame_number=show_frame_number) or 0)


def run_make_videos(argv):
    return int(run_render(argv) or 0)


def run_make_subtitles(*, archive_filters=None, title_filters=None, title_exact: bool = False):
    return int(
        run_render_subtitles(
            archive_filters=archive_filters,
            title_filters=title_filters,
            title_exact=title_exact,
        )
        or 0
    )


def run_generate_drive_checksum():
    return int(generate_drive_checksum() or 0)


def run_verify_drive(argv):
    return int(verify_drive(argv) or 0)


def run_make_comparisons(argv):
    return int(run_comparisons(argv) or 0)


def run_tuner(host: str = "0.0.0.0", port: int = 8092):
    run_tuner_server(host=host, port=int(port))
    return 0


def run_people_prefill(
    *,
    archive: str,
    chapter: str,
    cast_store: str,
    min_quality: float = 0.40,
    min_name_hits: int = 1,
    apply: bool = False,
    audit_file: str | None = None,
):
    result = prefill_people_from_cast(
        archive=str(archive or "").strip(),
        chapter_title=str(chapter or "").strip(),
        cast_store_dir=str(cast_store or "").strip(),
        min_quality=float(min_quality),
        min_name_hits=int(min_name_hits),
    )
    entries = list(result.entries or [])
    stats = dict(result.stats or {})

    print(
        "Cast prefill summary: "
        f"matched={int(stats.get('faces_matched', 0))}, "
        f"used={int(stats.get('faces_used', 0))}, "
        f"entries={int(stats.get('entries_generated', 0))}"
    )
    if entries:
        print("Generated entries:")
        for idx, row in enumerate(entries, start=1):
            print(
                f"{idx:02d}. {row.get('start')} - {row.get('end')} | {row.get('people')}"
            )
    else:
        print("No Cast matches found for this chapter.")

    if audit_file:
        audit_path = write_prefill_audit_tsv(Path(audit_file), entries)
        print(f"Audit TSV: {audit_path}")

    if apply:
        if not entries:
            print("Skip apply: no entries to write.")
            return 0
        people_path, written = apply_prefill_entries_to_people_tsv(
            archive=str(archive or "").strip(),
            chapter_title=str(chapter or "").strip(),
            entries=entries,
        )
        print(f"Wrote {int(written)} chapter-local row(s) to: {people_path}")
    else:
        print("Dry run only. Use --apply to write metadata/<archive>/people.tsv.")

    return 0

