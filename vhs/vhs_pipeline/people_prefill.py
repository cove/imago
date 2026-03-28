from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any

from common import ARCHIVE_DIR, METADATA_DIR, chapter_frame_bounds, parse_chapters, safe

FPS_NUM = 30000
FPS_DEN = 1001
PEOPLE_TSV_HEADER = "start\tend\tpeople"


def _normalize_token(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _parse_seconds(raw: Any) -> float | None:
    text = str(raw or "").strip().replace(",", ".")
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 1:
            sec = float(parts[0])
        elif len(parts) == 2:
            sec = float(int(parts[0]) * 60 + float(parts[1]))
        elif len(parts) == 3:
            sec = float(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))
        else:
            return None
    except Exception:
        return None
    if not (sec == sec):
        return None
    return max(0.0, float(sec))


def _to_timestamp(seconds: float) -> str:
    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    mins, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}.{int(ms):03d}"


def _parse_frame(raw: Any) -> int | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if not re.fullmatch(r"-?\d+", text):
        return None
    try:
        frame = int(text)
    except Exception:
        return None
    if frame < 0:
        return None
    return int(frame)


def _frame_to_seconds(frame: int) -> float:
    return float(int(frame) * FPS_DEN) / float(FPS_NUM)


def _parse_tsv_time_or_frame_seconds(raw: Any) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    # Backward compatibility: old files stored archive-global frames.
    if re.fullmatch(r"-?\d+", text):
        frame = _parse_frame(text)
        if frame is None:
            return None
        return _frame_to_seconds(frame)
    return _parse_seconds(text)


def _seconds_to_frame(seconds: float) -> int:
    return int(round(max(0.0, float(seconds)) * float(FPS_NUM) / float(FPS_DEN)))


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = str(raw or "").strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _chapter_boundary_seconds(chapter: dict[str, Any], boundary: str) -> float:
    raw_key = f"{boundary}_raw"
    raw_value = chapter.get(raw_key)
    tb_num = chapter.get("timebase_num")
    tb_den = chapter.get("timebase_den")
    if raw_value is not None and tb_num is not None and tb_den:
        try:
            return float(int(raw_value) * int(tb_num) / float(int(tb_den)))
        except Exception:
            pass
    return float(chapter.get(boundary, 0.0) or 0.0)


def _parse_timebase(raw: Any) -> tuple[int, int] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if "/" in text:
            num_s, den_s = text.split("/", 1)
            num = int(num_s.strip())
            den = int(den_s.strip())
        else:
            num = int(text)
            den = 1
    except Exception:
        return None
    if den == 0:
        return None
    if den < 0:
        num = -num
        den = -den
    return int(num), int(den)


def _load_chapter_context_from_tsv(archive: str, chapter_title: str) -> dict[str, Any] | None:
    chapters_path = METADATA_DIR / str(archive or "").strip() / "chapters.tsv"
    if not chapters_path.exists():
        return None
    target = str(chapter_title or "").strip()
    if not target:
        return None
    with chapters_path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for raw_row in reader:
            row = {str(k or "").strip().lower(): str(v or "").strip() for k, v in (raw_row or {}).items()}
            if str(row.get("title") or "").strip() != target:
                continue

            tb = _parse_timebase(row.get("timebase"))
            start_raw_text = str(row.get("start_raw") or row.get("start") or "").strip()
            end_raw_text = str(row.get("end_raw") or row.get("end") or "").strip()
            if not tb or not re.fullmatch(r"-?\d+", start_raw_text) or not re.fullmatch(r"-?\d+", end_raw_text):
                continue

            tb_num, tb_den = tb
            start_raw = int(start_raw_text)
            end_raw = int(end_raw_text)
            start_sec = float(Fraction(start_raw) * Fraction(tb_num, tb_den))
            end_sec = float(Fraction(end_raw) * Fraction(tb_num, tb_den))
            start_frame, end_frame = chapter_frame_bounds(
                {
                    "start_raw": start_raw,
                    "end_raw": end_raw,
                    "timebase_num": tb_num,
                    "timebase_den": tb_den,
                },
                fps_num=FPS_NUM,
                fps_den=FPS_DEN,
            )
            if end_sec <= start_sec or int(end_frame) <= int(start_frame):
                continue
            return {
                "archive": str(archive or "").strip(),
                "title": target,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "duration_sec": float(end_sec - start_sec),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
            }
    return None


def _load_chapter_context(archive: str, chapter_title: str) -> dict[str, Any]:
    target = str(chapter_title or "").strip()
    chapters_path = METADATA_DIR / str(archive or "").strip() / "chapters.ffmetadata"
    if chapters_path.exists():
        _ffm, chapters = parse_chapters(chapters_path)
        chapter = next(
            (row for row in chapters if str(row.get("title", "")).strip() == target),
            None,
        )
        if chapter is not None:
            start_sec = _chapter_boundary_seconds(chapter, "start")
            end_sec = _chapter_boundary_seconds(chapter, "end")
            start_frame, end_frame = chapter_frame_bounds(chapter, fps_num=FPS_NUM, fps_den=FPS_DEN)
            if end_sec <= start_sec:
                raise ValueError(f"Invalid chapter bounds for '{target}'.")
            if int(end_frame) <= int(start_frame):
                raise ValueError(f"Invalid chapter frame bounds for '{target}'.")
            return {
                "archive": str(archive or "").strip(),
                "title": target,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "duration_sec": float(end_sec - start_sec),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
            }

    tsv_context = _load_chapter_context_from_tsv(archive, target)
    if tsv_context is not None:
        return tsv_context

    if not chapters_path.exists():
        raise FileNotFoundError(f"Missing chapter metadata for archive '{archive}'.")
    raise ValueError(f"Unknown chapter title for archive '{archive}': {target}")


def _source_path_key(path: str | Path) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        try:
            p = p.resolve()
        except Exception:
            pass
    return str(p).replace("\\", "/").lower()


def _archive_source_keys(archive: str) -> set[str]:
    name = str(archive or "").strip()
    if not name:
        return set()
    return {
        _source_path_key(ARCHIVE_DIR / f"{name}.mkv"),
        _source_path_key(ARCHIVE_DIR / f"{name}_proxy.mp4"),
    }


def _classify_face_source(
    source_path: str,
    *,
    archive: str,
    chapter_title: str,
    archive_source_keys: set[str],
) -> str:
    key = _source_path_key(source_path)
    if key and key in archive_source_keys:
        return "archive"

    stem = Path(str(source_path or "")).stem
    stem_token = _normalize_token(stem)
    chapter_token = _normalize_token(chapter_title)
    archive_token = _normalize_token(archive)
    chapter_safe_token = _normalize_token(safe(chapter_title))

    if chapter_token and chapter_token in stem_token:
        return "chapter"
    if chapter_safe_token and chapter_safe_token in stem_token:
        return "chapter"
    if archive_token and archive_token in stem_token and ("proxy" in stem.lower() or "archive" in key):
        return "archive"
    return ""


def _estimate_step_seconds(
    times: list[float],
    *,
    default_step: float,
    min_step: float,
    max_step: float,
) -> float:
    if len(times) < 2:
        return float(default_step)
    deltas = [float(b - a) for a, b in zip(times, times[1:]) if float(b - a) > 0.05]
    if not deltas:
        return float(default_step)
    step = float(median(deltas))
    step = max(float(min_step), min(float(max_step), step))
    return step


def _read_people_tsv_rows(path: Path) -> list[tuple[float, float, str]]:
    rows: list[tuple[float, float, str]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = str(raw or "").strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower.startswith("start_frame\t") or lower.startswith("start_frame,end_frame"):
            continue
        if lower.startswith("start\t") or lower.startswith("start,end"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(",")
        if len(parts) < 3:
            continue
        start = _parse_tsv_time_or_frame_seconds(parts[0])
        end = _parse_tsv_time_or_frame_seconds(parts[1])
        people = re.sub(r"\s+", " ", ",".join(parts[2:]).strip())
        if start is None or end is None or not people:
            continue
        if float(end) <= float(start):
            if abs(float(end) - float(start)) < 1e-9:
                end = float(start) + _frame_to_seconds(1)
            else:
                continue
        rows.append((float(start), float(end), str(people)))
    return rows


def _canonicalize_people_tsv_rows(
    rows: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    cleaned: list[tuple[float, float, str]] = []
    for start, end, people in list(rows or []):
        try:
            a = float(start)
            b = float(end)
        except Exception:
            continue
        if float(b) <= float(a):
            continue
        text = re.sub(r"\s+", " ", str(people or "")).strip()
        if not text:
            continue
        cleaned.append((max(0.0, float(a)), max(0.0, float(b)), text))
    if not cleaned:
        return []

    cleaned.sort(key=lambda item: (item[0], item[1], item[2].lower()))
    out: list[tuple[float, float, str]] = []
    for start, end, people in cleaned:
        if out:
            prev_start, prev_end, prev_people = out[-1]
            if prev_people == people and float(prev_end) + 0.001 >= float(start):
                out[-1] = (prev_start, max(float(prev_end), float(end)), prev_people)
                continue
        out.append((round(float(start), 3), round(float(end), 3), people))
    return out


def _write_people_tsv_rows(path: Path, rows: list[tuple[float, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [PEOPLE_TSV_HEADER]
    for start, end, people in list(rows or []):
        lines.append(f"{_to_timestamp(float(start))}\t{_to_timestamp(float(end))}\t{people}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class PrefillResult:
    entries: list[dict[str, Any]]
    stats: dict[str, Any]
    chapter_start_sec: float
    chapter_end_sec: float


def prefill_people_from_cast(
    *,
    archive: str,
    chapter_title: str,
    cast_store_dir: str | Path,
    min_quality: float = 0.40,
    min_name_hits: int = 1,
    default_step_seconds: float = 2.0,
    min_step_seconds: float = 0.75,
    max_step_seconds: float = 8.0,
    min_segment_seconds: float = 0.75,
    merge_gap_seconds: float = 0.35,
) -> PrefillResult:
    chapter = _load_chapter_context(archive, chapter_title)
    chapter_start = float(chapter["start_sec"])
    chapter_end = float(chapter["end_sec"])
    chapter_duration = float(chapter["duration_sec"])

    cast_root = Path(cast_store_dir)
    people_payload = _read_json(cast_root / "people.json", default={"people": []})
    people_rows = people_payload.get("people", []) if isinstance(people_payload, dict) else []
    faces_rows = _read_jsonl(cast_root / "faces.jsonl")

    people_by_id: dict[str, str] = {}
    for row in list(people_rows or []):
        if not isinstance(row, dict):
            continue
        person_id = str(row.get("person_id") or "").strip()
        display_name = re.sub(r"\s+", " ", str(row.get("display_name") or "").strip())
        if person_id and display_name:
            people_by_id[person_id] = display_name

    archive_keys = _archive_source_keys(archive)
    observations: list[tuple[float, str]] = []
    source_counts = {"chapter": 0, "archive": 0}
    matched_faces = 0
    vhs_faces = 0

    for row in list(faces_rows or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("source_type") or "").strip().lower() != "vhs":
            continue
        vhs_faces += 1
        person_id = str(row.get("person_id") or "").strip()
        if not person_id:
            continue
        person_name = people_by_id.get(person_id)
        if not person_name:
            continue
        timestamp = _parse_seconds(row.get("timestamp"))
        if timestamp is None:
            continue
        raw_quality = row.get("quality")
        if raw_quality is not None:
            try:
                if float(raw_quality) < float(min_quality):
                    continue
            except Exception:
                continue

        source_path = str(row.get("source_path") or "").strip()
        mode = _classify_face_source(
            source_path,
            archive=archive,
            chapter_title=chapter_title,
            archive_source_keys=archive_keys,
        )
        if not mode:
            continue
        if mode == "archive":
            local_sec = float(timestamp) - chapter_start
        else:
            local_sec = float(timestamp)
        if local_sec < 0.0 or local_sec > chapter_duration:
            continue

        matched_faces += 1
        source_counts[mode] = int(source_counts.get(mode, 0)) + 1
        observations.append((round(float(local_sec), 3), person_name))

    if not observations:
        return PrefillResult(
            entries=[],
            stats={
                "faces_total": len(faces_rows),
                "faces_vhs": int(vhs_faces),
                "faces_matched": 0,
                "entries_generated": 0,
                "source_counts": source_counts,
            },
            chapter_start_sec=chapter_start,
            chapter_end_sec=chapter_end,
        )

    person_hits: dict[str, int] = {}
    for _ts, name in observations:
        person_hits[name] = int(person_hits.get(name, 0)) + 1
    keep_names = {name for name, hits in person_hits.items() if int(hits) >= max(1, int(min_name_hits))}
    observations = [(ts, name) for ts, name in observations if name in keep_names]
    if not observations:
        return PrefillResult(
            entries=[],
            stats={
                "faces_total": len(faces_rows),
                "faces_vhs": int(vhs_faces),
                "faces_matched": int(matched_faces),
                "entries_generated": 0,
                "source_counts": source_counts,
            },
            chapter_start_sec=chapter_start,
            chapter_end_sec=chapter_end,
        )

    by_time: dict[float, dict[str, int]] = {}
    for ts, name in observations:
        bucket = by_time.setdefault(float(ts), {})
        bucket[name] = int(bucket.get(name, 0)) + 1

    times = sorted(by_time.keys())
    step = _estimate_step_seconds(
        times,
        default_step=float(default_step_seconds),
        min_step=float(min_step_seconds),
        max_step=float(max_step_seconds),
    )
    half_window = max(0.15, step * 0.45)
    min_seg = max(0.25, float(min_segment_seconds))
    merge_gap = max(0.0, float(merge_gap_seconds))

    rows: list[dict[str, Any]] = []
    for ts in times:
        counts = by_time[ts]
        ordered_names = sorted(
            counts.keys(),
            key=lambda n: (
                -int(counts.get(n, 0)),
                -int(person_hits.get(n, 0)),
                str(n).lower(),
            ),
        )
        people = " | ".join(ordered_names)
        if not people:
            continue
        start = max(0.0, float(ts) - half_window)
        end = min(chapter_duration, float(ts) + half_window)
        if end - start < min_seg:
            center = float(ts)
            start = max(0.0, center - (min_seg * 0.5))
            end = min(chapter_duration, center + (min_seg * 0.5))
        rows.append(
            {
                "start": float(start),
                "end": float(end),
                "people": people,
            }
        )

    merged: list[dict[str, Any]] = []
    for row in rows:
        start = float(row["start"])
        end = float(row["end"])
        people = str(row["people"])
        if end <= start:
            continue
        if merged:
            prev = merged[-1]
            if str(prev["people"]) == people and start <= float(prev["end"]) + merge_gap:
                prev["end"] = max(float(prev["end"]), end)
                continue
        merged.append({"start": start, "end": end, "people": people})

    entries: list[dict[str, Any]] = []
    for row in merged:
        start = round(float(row["start"]), 3)
        end = round(float(row["end"]), 3)
        if end <= start:
            continue
        entries.append(
            {
                "start_seconds": float(start),
                "end_seconds": float(end),
                "start": _to_timestamp(start),
                "end": _to_timestamp(end),
                "people": str(row["people"]),
            }
        )

    return PrefillResult(
        entries=entries,
        stats={
            "faces_total": len(faces_rows),
            "faces_vhs": int(vhs_faces),
            "faces_matched": int(matched_faces),
            "faces_used": int(len(observations)),
            "entries_generated": int(len(entries)),
            "unique_people": int(len(keep_names)),
            "sample_step_seconds": float(round(step, 3)),
            "source_counts": source_counts,
        },
        chapter_start_sec=chapter_start,
        chapter_end_sec=chapter_end,
    )


def apply_prefill_entries_to_people_tsv(
    *,
    archive: str,
    chapter_title: str,
    entries: list[dict[str, Any]],
) -> tuple[Path, int]:
    chapter = _load_chapter_context(archive, chapter_title)
    chapter_start_sec = float(chapter["start_sec"])
    chapter_end_sec = float(chapter["end_sec"])
    chapter_len_sec = max(_frame_to_seconds(1), float(chapter_end_sec) - float(chapter_start_sec))
    tsv_path = METADATA_DIR / str(archive or "").strip() / "people.tsv"

    existing = _read_people_tsv_rows(tsv_path)
    kept: list[tuple[float, float, str]] = []
    for start, end, people in existing:
        if float(end) <= float(chapter_start_sec) or float(start) >= float(chapter_end_sec):
            kept.append((float(start), float(end), str(people)))
            continue
        if float(start) < float(chapter_start_sec):
            kept.append((float(start), float(chapter_start_sec), str(people)))
        if float(end) > float(chapter_end_sec):
            kept.append((float(chapter_end_sec), float(end), str(people)))

    chapter_rows: list[tuple[float, float, str]] = []
    for item in list(entries or []):
        start_local = _parse_seconds(item.get("start_seconds", item.get("start")))
        end_local = _parse_seconds(item.get("end_seconds", item.get("end")))
        people = re.sub(r"\s+", " ", str(item.get("people") or "").strip())
        if start_local is None or end_local is None or end_local <= start_local or not people:
            continue
        local_start_sec = max(0.0, min(float(chapter_len_sec), float(start_local)))
        local_end_sec = max(0.0, min(float(chapter_len_sec), float(end_local)))
        if local_end_sec <= local_start_sec:
            if local_start_sec >= float(chapter_len_sec):
                continue
            local_end_sec = min(float(chapter_len_sec), float(local_start_sec) + _frame_to_seconds(1))
        chapter_rows.append(
            (
                float(chapter_start_sec) + float(local_start_sec),
                float(chapter_start_sec) + float(local_end_sec),
                str(people),
            )
        )

    merged = _canonicalize_people_tsv_rows([*kept, *chapter_rows])
    _write_people_tsv_rows(tsv_path, merged)
    return tsv_path, int(len(chapter_rows))


def write_prefill_audit_tsv(path: Path, entries: list[dict[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["start\tend\tpeople"]
    for row in list(entries or []):
        lines.append(f"{str(row.get('start') or '')}\t{str(row.get('end') or '')}\t{str(row.get('people') or '')}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
