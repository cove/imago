#!/usr/bin/env python3.11
from __future__ import annotations

import base64
import binascii
import bisect
import cProfile
import json
import html
import importlib
import os
import re
import csv
import io
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import wave
from collections import OrderedDict
from dataclasses import dataclass, field
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse
from contextlib import redirect_stdout

import cv2
import numpy as np
from PIL import Image, ImageOps

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import (
    ARCHIVE_DIR,
    CLIPS_DIR,
    FFMPEG_BIN,
    METADATA_DIR,
    VIDEOS_DIR,
    WHISPER_MODEL_DIR,
    combined_score,
    compute_threshold,
    get_audio_sync_offset_for_chapter,
    get_gamma_profile_for_chapter,
    get_transcript_mode_for_chapter,
    parse_chapters,
    safe,
    update_chapter_audio_sync_in_render_settings,
    update_chapter_gamma_in_render_settings,
    update_chapter_transcript_in_chapters_tsv,
)
from libs.vhs_tuner_core import (
    _bgr_to_jpeg_b64,
    _chapter_bad_overrides,
    _chapter_extract_cache_path,
    _ensure_render_chapter_extract,
    _env_truthy,
    _find_chapter,
    _get_archives,
    _normalize_frame_span,
    _resolve_archive_video,
    build_archive_state,
    estimate_gamma_from_frames,
    extract_frames,
    find_spike_regions,
    load_cached_signals,
    persist_bad_frames_for_chapter,
    slugify,
    suggest_iqr_k,
    RENDER_DEBUG_EXTRACT_FRAME_NUMBERS_ENV,
    TUNER_DEBUG_EXTRACT_ENV,
)
from vhs_pipeline.people_prefill import prefill_people_from_cast
from vhs_pipeline.metadata import ffmetadata_to_chapters_tsv
from vhs_pipeline.render_pipeline import (
    make_extract_audio,
    subtitle_entries_from_whisper_result,
    whisper_transcribe,
)

try:
    import whisper
except Exception:
    whisper = None

STATIC_DIR = _HERE / "static"
INDEX_HTML = STATIC_DIR / "index.html"
_STATIC_FILE_CACHE: dict[str, tuple[str, bytes]] = {}  # path -> (content_type, bytes)
SESSION_COOKIE = "vhs_plain_wizard_sid"
FPS_NUM = 30000
FPS_DEN = 1001
PEOPLE_TSV_HEADER = "start\tend\tpeople"
SUBTITLES_TSV_HEADER = "start\tend\ttext\tspeaker\tconfidence\tsource"
TSV_META_CHAPTER_INDEX_COL = "__chapter_index"
TSV_FFMETA_PREFIX = "ffmeta_"
TSV_META_PREFIX = "__"
CHAPTERS_TSV_META_COLUMNS = [
    TSV_META_CHAPTER_INDEX_COL,
]
CHAPTER_FFMETADATA_COMPUTED_KEYS = {
    "start_raw",
    "end_raw",
    "timebase_num",
    "timebase_den",
}
DEFAULT_CAST_STORE_DIR = (PROJECT_ROOT.parent / "cast" / "data").resolve()
WHISPER_MODEL_NAME = "turbo"
CONTACT_SHEET_TILE_WIDTH = 160
CONTACT_SHEET_TILE_HEIGHT = 120
CONTACT_SHEET_CHUNK_SIZE = 512
CONTACT_SHEET_COLUMNS = 8
CONTACT_SHEET_CACHE_LIMIT = 64


@dataclass
class SessionState:
    archive: str = ""
    chapter: str = ""
    chapters: list[dict[str, Any]] = field(default_factory=list)
    chapter_rows: list[list[Any]] = field(default_factory=list)

    start_frame: int = 0
    end_frame: int = 1
    debug_extract: bool = False

    wc: float = 0.25
    wn: float = 0.25
    wt: float = 0.25
    ww: float = 0.25
    t_mode: str = "iqr"
    iqr_k: float = 3.5
    tval: float = 1.0
    bpct: float = 10.0

    fids: list[int] = field(default_factory=list)
    b64: list[str] = field(default_factory=list)
    sigs: dict[str, np.ndarray] = field(default_factory=dict)
    overrides: dict[int, str] = field(default_factory=dict)
    force_all_frames_good: bool = False
    threshold: float = 0.0
    load_running: bool = False
    load_progress: float = 0.0
    load_message: str = ""
    load_sample_done: int = 0
    load_sample_total: int = 0
    load_cancel_requested: bool = False
    load_meta_ready: bool = False
    preview_running: bool = False
    preview_progress: float = 0.0
    preview_message: str = ""
    preview_frame_done: int = 0
    preview_frame_total: int = 0
    preview_video_path: str = ""
    chapter_audio_path: str = ""
    chapter_audio_key: str = ""
    subtitles_running: bool = False
    subtitles_progress: float = 0.0
    subtitles_message: str = ""
    subtitles_segment_done: int = 0
    subtitles_segment_total: int = 0
    subtitles_cancel_requested: bool = False
    gamma_default: float = 1.0
    gamma_ranges: list[dict[str, Any]] = field(default_factory=list)
    audio_sync_offset: float = 0.0
    audio_sync_audio_path: str = ""
    audio_sync_audio_key: str = ""
    people_entries: list[dict[str, Any]] = field(default_factory=list)
    subtitle_entries: list[dict[str, Any]] = field(default_factory=list)
    auto_transcript: str = "off"
    split_entries: list[dict[str, Any]] = field(default_factory=list)
    partial_fids: list[int] = field(default_factory=list)
    partial_b64: list[str] = field(default_factory=list)
    partial_sigs: dict[str, list[float]] = field(
        default_factory=lambda: {"chroma": [], "noise": [], "tear": [], "wave": []}
    )
    frame_source_video_path: str = ""
    frame_source_read_offset: int = 0
    # Persistent VideoCapture reused across contact-sheet requests for the same video.
    # Access only while holding _video_cap_lock.
    _video_cap: Any = field(default=None, repr=False, compare=False)
    _video_cap_path: str = field(default="", repr=False, compare=False)
    _video_cap_last_fid: int = field(default=-1, repr=False, compare=False)
    _video_cap_lock: Any = field(
        default_factory=threading.Lock, repr=False, compare=False
    )


def _close_session_video_cap(session: "SessionState") -> None:
    """Release the session's cached VideoCapture if open. Call when switching videos."""
    with session._video_cap_lock:
        if session._video_cap is not None:
            session._video_cap.release()
            session._video_cap = None
            session._video_cap_path = ""
            session._video_cap_last_fid = -1


_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[str, SessionState] = {}
_FRAME_CONTACT_SHEET_CACHE_LOCK = threading.Lock()
_FRAME_CONTACT_SHEET_CACHE: OrderedDict[str, tuple[str, bytes]] = OrderedDict()
_SIGNALS_MEMO_LOCK = threading.Lock()
_SIGNALS_MEMO_KEY: tuple | None = None
_SIGNALS_MEMO_VAL: tuple | None = None
_PROF_LOCK = threading.Lock()
_WHISPER_MODEL_LOCK = threading.Lock()
_WHISPER_MODEL: Any | None = None
_WHISPER_TQDM_PATCH_LOCK = threading.Lock()
_WHISPER_TRANSCRIBE_MODULE: Any | None = None


class _SubtitlesCancelledError(Exception):
    pass


def _set_load_progress(
    session: SessionState,
    *,
    running: bool | None = None,
    progress: float | None = None,
    message: str | None = None,
    sample_done: int | None = None,
    sample_total: int | None = None,
) -> None:
    if running is not None:
        session.load_running = bool(running)
    if progress is not None:
        session.load_progress = max(0.0, min(100.0, float(progress)))
    if message is not None:
        session.load_message = str(message)
    if sample_done is not None:
        session.load_sample_done = max(0, int(sample_done))
    if sample_total is not None:
        session.load_sample_total = max(0, int(sample_total))


def _set_preview_progress(
    session: SessionState,
    *,
    running: bool | None = None,
    progress: float | None = None,
    message: str | None = None,
    frame_done: int | None = None,
    frame_total: int | None = None,
) -> None:
    if running is not None:
        session.preview_running = bool(running)
    if progress is not None:
        session.preview_progress = max(0.0, min(100.0, float(progress)))
    if message is not None:
        session.preview_message = str(message)
    if frame_done is not None:
        session.preview_frame_done = max(0, int(frame_done))
    if frame_total is not None:
        session.preview_frame_total = max(0, int(frame_total))


def _set_subtitles_progress(
    session: SessionState,
    *,
    running: bool | None = None,
    progress: float | None = None,
    message: str | None = None,
    segment_done: int | None = None,
    segment_total: int | None = None,
) -> None:
    if running is not None:
        session.subtitles_running = bool(running)
    if progress is not None:
        session.subtitles_progress = max(0.0, min(100.0, float(progress)))
    if message is not None:
        session.subtitles_message = str(message)
    if segment_done is not None:
        session.subtitles_segment_done = max(0, int(segment_done))
    if segment_total is not None:
        session.subtitles_segment_total = max(0, int(segment_total))


def _normalize_iqr_k(raw: Any, default: float = 3.5) -> float:
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    return max(0.0, min(12.0, float(value)))


def _normalize_payload_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _normalize_gamma_value(raw: Any, default: float = 1.0) -> float:
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    if not (value == value):
        value = float(default)
    return max(0.05, min(8.0, float(value)))


def _normalize_gamma_ranges_payload(
    raw_ranges: Any,
    *,
    ch_start: int | None = None,
    ch_end: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[tuple[int, int, float, int]] = []
    for idx, item in enumerate(list(raw_ranges or [])):
        start = end = gamma = None
        if isinstance(item, dict):
            start = item.get("start_frame")
            end = item.get("end_frame")
            gamma = item.get("gamma")
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start, end, gamma = item[0], item[1], item[2]
        try:
            a = int(start)
            b = int(end)
        except Exception:
            continue
        if b <= a:
            continue
        if ch_start is not None and ch_end is not None:
            a = max(int(ch_start), a)
            b = min(int(ch_end), b)
            if b <= a:
                continue
        g = _normalize_gamma_value(gamma, default=1.0)
        rows.append((a, b, g, idx))
    if not rows:
        return []

    boundaries = set()
    for a, b, _g, _idx in rows:
        boundaries.add(int(a))
        boundaries.add(int(b))
    cuts = sorted(boundaries)
    out: list[tuple[int, int, float]] = []
    for i in range(len(cuts) - 1):
        seg_a = int(cuts[i])
        seg_b = int(cuts[i + 1])
        if seg_b <= seg_a:
            continue
        winner_idx = -1
        winner_gamma = None
        for a, b, g, idx in rows:
            if a <= seg_a and seg_b <= b and idx >= winner_idx:
                winner_idx = idx
                winner_gamma = float(g)
        if winner_gamma is None:
            continue
        if (
            out
            and out[-1][1] == seg_a
            and abs(float(out[-1][2]) - float(winner_gamma)) < 1e-6
        ):
            prev_a, _prev_b, prev_g = out[-1]
            out[-1] = (prev_a, seg_b, prev_g)
        else:
            out.append((seg_a, seg_b, float(winner_gamma)))
    return [
        {"start_frame": int(a), "end_frame": int(b), "gamma": round(float(g), 4)}
        for a, b, g in out
        if int(b) > int(a)
    ]


def _frame_to_seconds(frame_id: int) -> float:
    return float(int(frame_id) * FPS_DEN) / float(FPS_NUM)


def _seconds_to_frame(seconds: float) -> int:
    return int(round(max(0.0, float(seconds)) * float(FPS_NUM) / float(FPS_DEN)))


def _seconds_to_timestamp(seconds: float) -> str:
    secs = max(0.0, float(seconds))
    total_ms = int(round(secs * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    whole_seconds, millis = divmod(rem, 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(whole_seconds):02d}.{int(millis):03d}"


def _parse_timestamp_seconds(raw: Any) -> float | None:
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    text = text.replace(",", ".")
    parts = text.split(":")
    try:
        if len(parts) == 1:
            value = float(parts[0])
        elif len(parts) == 2:
            mins = int(parts[0])
            secs = float(parts[1])
            value = float((mins * 60) + secs)
        elif len(parts) == 3:
            hours = int(parts[0])
            mins = int(parts[1])
            secs = float(parts[2])
            value = float((hours * 3600) + (mins * 60) + secs)
        else:
            return None
    except Exception:
        return None
    if not (value == value):
        return None
    return max(0.0, float(value))


def _normalize_people_entries_payload(
    raw_entries: Any,
    *,
    chapter_duration_seconds: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[tuple[float, float, str, int]] = []
    duration = None
    if chapter_duration_seconds is not None:
        try:
            duration = max(0.0, float(chapter_duration_seconds))
        except Exception:
            duration = None

    for idx, item in enumerate(list(raw_entries or [])):
        start_raw = end_raw = people_raw = None
        if isinstance(item, dict):
            start_raw = item.get("start_seconds", item.get("start"))
            end_raw = item.get("end_seconds", item.get("end"))
            people_raw = item.get("people")
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start_raw, end_raw, people_raw = item[0], item[1], item[2]
        start = _parse_timestamp_seconds(start_raw)
        end = _parse_timestamp_seconds(end_raw)
        if start is None or end is None or end <= start:
            continue
        if duration is not None:
            start = max(0.0, min(duration, float(start)))
            end = max(0.0, min(duration, float(end)))
            if end <= start:
                continue
        people = re.sub(r"\s+", " ", str(people_raw or "")).strip()
        if not people:
            continue
        rows.append((float(start), float(end), people, int(idx)))

    if not rows:
        return []

    rows.sort(key=lambda item: (item[0], item[1], item[3]))
    out: list[dict[str, Any]] = []
    for start, end, people, _idx in rows:
        start_s = round(float(start), 3)
        end_s = round(float(end), 3)
        if out:
            prev = out[-1]
            if (
                str(prev["people"]) == people
                and float(prev["end_seconds"]) + 0.001 >= start_s
            ):
                prev["end_seconds"] = max(float(prev["end_seconds"]), end_s)
                prev["end"] = _seconds_to_timestamp(float(prev["end_seconds"]))
                continue
        out.append(
            {
                "start_seconds": start_s,
                "end_seconds": end_s,
                "start": _seconds_to_timestamp(start_s),
                "end": _seconds_to_timestamp(end_s),
                "people": people,
            }
        )
    return out


def _normalize_subtitle_optional_text(raw: Any) -> str:
    return re.sub(r"\s+", " ", str(raw if raw is not None else "")).strip()


def _parse_subtitle_confidence(raw: Any) -> float | None:
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except Exception:
        return None
    if not (value == value):
        return None
    return max(0.0, min(1.0, float(value)))


def _format_subtitle_confidence(raw: Any) -> str:
    parsed = _parse_subtitle_confidence(raw)
    if parsed is None:
        return ""
    return f"{parsed:.4f}".rstrip("0").rstrip(".")


def _normalize_subtitle_entries_payload(
    raw_entries: Any,
    *,
    chapter_duration_seconds: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[tuple[float, float, str, str, float | None, str, int]] = []
    duration = None
    if chapter_duration_seconds is not None:
        try:
            duration = max(0.0, float(chapter_duration_seconds))
        except Exception:
            duration = None

    for idx, item in enumerate(list(raw_entries or [])):
        start_raw = end_raw = text_raw = None
        speaker_raw = confidence_raw = source_raw = None
        if isinstance(item, dict):
            start_raw = item.get("start_seconds", item.get("start"))
            end_raw = item.get("end_seconds", item.get("end"))
            text_raw = item.get("text")
            speaker_raw = item.get("speaker")
            confidence_raw = item.get("confidence")
            source_raw = item.get("source")
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start_raw, end_raw, text_raw = item[0], item[1], item[2]
            if len(item) >= 4:
                speaker_raw = item[3]
            if len(item) >= 5:
                confidence_raw = item[4]
            if len(item) >= 6:
                source_raw = item[5]
        start = _parse_timestamp_seconds(start_raw)
        end = _parse_timestamp_seconds(end_raw)
        if start is None or end is None or end <= start:
            continue
        if duration is not None:
            start = max(0.0, min(duration, float(start)))
            end = max(0.0, min(duration, float(end)))
            if end <= start:
                continue
        text = _normalize_subtitle_optional_text(text_raw)
        if not text:
            continue
        speaker = _normalize_subtitle_optional_text(speaker_raw)
        confidence = _parse_subtitle_confidence(confidence_raw)
        source = _normalize_subtitle_optional_text(source_raw)
        rows.append(
            (float(start), float(end), text, speaker, confidence, source, int(idx))
        )

    if not rows:
        return []

    rows.sort(key=lambda item: (item[0], item[1], item[6]))
    out: list[dict[str, Any]] = []
    for start, end, text, speaker, confidence, source, _idx in rows:
        start_s = round(float(start), 3)
        end_s = round(float(end), 3)
        out.append(
            {
                "start_seconds": start_s,
                "end_seconds": end_s,
                "start": _seconds_to_timestamp(start_s),
                "end": _seconds_to_timestamp(end_s),
                "text": text,
                "speaker": speaker,
                "confidence": confidence,
                "source": source,
            }
        )
    return out


def _chapters_ffmetadata_path(archive: str) -> Path:
    return METADATA_DIR / str(archive or "").strip() / "chapters.ffmetadata"


def _chapters_tsv_path(archive: str) -> Path:
    return METADATA_DIR / str(archive or "").strip() / "chapters.tsv"


def _rename_chapter_outputs(old_title: str, new_title: str) -> list[str]:
    """Rename rendered output files and remux MP4 to update embedded title metadata."""
    old_safe = safe(old_title)
    new_safe = safe(new_title)
    renamed: list[str] = []
    for output_dir in [VIDEOS_DIR, CLIPS_DIR]:
        if not output_dir:
            continue
        old_mp4 = output_dir / f"{old_safe}.mp4"
        if not old_mp4.exists() or old_mp4.stat().st_size <= 100_000:
            continue
        # Rename subtitle sidecars
        for ext in (".srt", ".vtt", ".ass"):
            old_f = output_dir / f"{old_safe}{ext}"
            new_f = output_dir / f"{new_safe}{ext}"
            if old_f.exists():
                old_f.rename(new_f)
                renamed.append(new_f.name)
        # Remux MP4 to update embedded title metadata (stream copy, no re-encode)
        new_mp4 = output_dir / f"{new_safe}.mp4"
        tmp_mp4 = output_dir / f"{new_safe}_renametmp.mp4"
        result = subprocess.run(
            [
                str(FFMPEG_BIN),
                "-y",
                "-i",
                str(old_mp4),
                "-c",
                "copy",
                "-metadata",
                f"title={new_title}",
                str(tmp_mp4),
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            old_mp4.unlink()
            tmp_mp4.rename(new_mp4)
        else:
            tmp_mp4.unlink(missing_ok=True)
            old_mp4.rename(new_mp4)
        renamed.append(new_mp4.name)
        # Rename temp dir if it still exists
        old_temp = output_dir / f"{old_safe}_temp"
        new_temp = output_dir / f"{new_safe}_temp"
        if old_temp.exists() and not new_temp.exists():
            old_temp.rename(new_temp)
        break
    return renamed


def _normalize_split_entries_payload(
    raw_entries: Any,
    *,
    chapter_frame_count: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[tuple[int, int, str, int]] = []
    frame_cap: int | None = None
    if chapter_frame_count is not None:
        try:
            frame_cap = max(0, int(chapter_frame_count))
        except Exception:
            frame_cap = None

    for idx, item in enumerate(list(raw_entries or [])):
        start_raw = end_raw = title_raw = None
        if isinstance(item, dict):
            start_raw = item.get("start_frame", item.get("start"))
            end_raw = item.get("end_frame", item.get("end"))
            title_raw = item.get("title", item.get("text"))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start_raw, end_raw, title_raw = item[0], item[1], item[2]

        start = _parse_frame_value(start_raw)
        end = _parse_frame_value(end_raw)
        if start is None or end is None or end <= start:
            continue
        if frame_cap is not None:
            start = max(0, min(int(frame_cap), int(start)))
            end = max(0, min(int(frame_cap), int(end)))
            if end <= start:
                continue
        title = _normalize_subtitle_optional_text(title_raw)
        if not title:
            continue
        rows.append((int(start), int(end), title, int(idx)))

    if not rows:
        return []

    rows.sort(key=lambda item: (item[0], item[1], item[3]))
    out: list[dict[str, Any]] = []
    for start_frame, end_frame, title, _idx in rows:
        out.append(
            {
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start": str(int(start_frame)),
                "end": str(int(end_frame)),
                "title": str(title),
            }
        )
    return out


def _default_split_entries_for_chapter(
    chapter_title: str, chapter_frame_count: int
) -> list[dict[str, Any]]:
    frame_count = max(1, int(chapter_frame_count))
    title = _normalize_subtitle_optional_text(chapter_title) or "Chapter 1"
    return [
        {
            "start_frame": 0,
            "end_frame": int(frame_count),
            "start": "0",
            "end": str(int(frame_count)),
            "title": title,
        }
    ]


def _read_chapters_tsv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    p = Path(path)
    if not p.exists():
        return [], []
    with p.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        raw_header = list(reader.fieldnames or [])
        header: list[str] = []
        seen: set[str] = set()
        for col in raw_header:
            key = str(col or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            header.append(key)
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            row: dict[str, str] = {}
            has_any = False
            for col in header:
                value = str((raw_row or {}).get(col, "") or "").strip()
                row[col] = value
                if value:
                    has_any = True
            if has_any:
                rows.append(row)
    return header, rows


def _write_chapters_tsv_rows(
    path: Path, columns: list[str], rows: list[dict[str, Any]]
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header: list[str] = []
    seen: set[str] = set()
    for col in list(columns or []):
        key = str(col or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        header.append(key)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=header,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for raw_row in list(rows or []):
            row = {col: str((raw_row or {}).get(col, "") or "") for col in header}
            writer.writerow(row)


def _row_ci_get(row: dict[str, Any], *names: str) -> Any:
    lowered: dict[str, Any] = {}
    for key, value in dict(row or {}).items():
        text = str(key or "").strip().lower()
        if text and text not in lowered:
            lowered[text] = value
    for name in names:
        key = str(name or "").strip().lower()
        if key in lowered:
            return lowered[key]
    return None


def _chapter_row_bounds(row: dict[str, Any]) -> tuple[int | None, int | None]:
    start_raw = _row_ci_get(row, "start", "start_frame")
    end_raw = _row_ci_get(row, "end", "end_frame")
    return _parse_frame_value(start_raw), _parse_frame_value(end_raw)


def _chapter_row_title(row: dict[str, Any]) -> str:
    return _normalize_subtitle_optional_text(
        _row_ci_get(row, "title", "split_title", "chapter_title", "chaptertitle")
    )


def _chapter_row_matches(
    row: dict[str, Any], chapter_title: str, chapter_start: int
) -> bool:
    """True when a TSV row represents the given chapter (matched by title + start_frame).

    Chapters are flat, independent ranges in the archive — there is no parent-child
    relationship. A row is considered "the same chapter" when title and start_frame
    both match, regardless of end_frame (which may have been updated).
    """
    row_start, _row_end = _chapter_row_bounds(row)
    row_title = _chapter_row_title(row)
    return (
        row_start is not None
        and int(row_start) == int(chapter_start)
        and row_title == str(chapter_title or "").strip()
    )


def _chapter_order_keys_for_row(header: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in list(header or []):
        key = str(col or "").strip()
        if (
            not key
            or key == "parent_chapter"
            or key.startswith(TSV_META_PREFIX)
            or key.startswith(TSV_FFMETA_PREFIX)
            or key in seen
        ):
            continue
        seen.add(key)
        out.append(key)
    return out


def _canonical_chapters_base(
    path: Path, archive_name: str
) -> tuple[list[str], list[dict[str, str]]]:
    header, rows = _read_chapters_tsv_rows(path)
    if rows:
        return header, rows
    ffmeta_path = _chapters_ffmetadata_path(archive_name)
    if not ffmeta_path.exists():
        return header, rows
    with tempfile.TemporaryDirectory(prefix="chapters_tsv_base_") as td:
        tmp_path = Path(td) / "chapters.tsv"
        with redirect_stdout(io.StringIO()):
            ffmetadata_to_chapters_tsv(ffmeta_path, tmp_path)
        return _read_chapters_tsv_rows(tmp_path)


def _build_chapter_row_from_template(
    template_row: dict[str, Any],
    header: list[str],
    global_start: int,
    global_end: int,
    title: str,
) -> dict[str, Any]:
    """Build a new chapter row by copying a template row and updating start, end, and title.

    Chapters are flat, independent ranges — this simply creates a new row with the
    correct bounds and title, inheriting metadata fields (archive name, dates, etc.)
    from the template.
    """
    row = dict(template_row or {})
    chapter_keys = _chapter_order_keys_for_row(header)
    for key in chapter_keys:
        lowered = str(key or "").strip().lower()
        if lowered == "start":
            row[key] = str(int(global_start))
        elif lowered == "end":
            row[key] = str(int(global_end))
        elif lowered == "title":
            row[key] = str(title)
    if not any(str(key or "").strip().lower() == "title" for key in chapter_keys):
        row["title"] = str(title)
    row[TSV_META_CHAPTER_INDEX_COL] = ""
    return row


def _reindex_canonical_chapter_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(list(rows or []), start=1):
        row = dict(raw or {})
        row[TSV_META_CHAPTER_INDEX_COL] = str(int(idx))
        out.append(row)
    return out


def _chapters_ffmetadata_context(
    archive: str,
    chapter_title: str,
) -> tuple[dict[str, Any], list[str], list[str], dict[str, Any] | None]:
    ffmetadata: dict[str, Any] = {}
    chapters: list[dict[str, Any]] = []
    ffmeta_path = _chapters_ffmetadata_path(archive)
    if ffmeta_path.exists():
        try:
            ffmetadata, chapters = parse_chapters(ffmeta_path)
        except Exception:
            ffmetadata, chapters = {}, []

    global_fields: list[str] = []
    seen_global: set[str] = set()
    for key in list(ffmetadata.keys()):
        col = str(key or "").strip().lower()
        if not col or col in seen_global:
            continue
        seen_global.add(col)
        global_fields.append(col)

    chapter_fields: list[str] = []
    seen_chapter: set[str] = set()
    for ch in list(chapters or []):
        if not isinstance(ch, dict):
            continue
        for key in list(ch.keys()):
            col = str(key or "").strip().lower()
            if (
                not col
                or col in CHAPTER_FFMETADATA_COMPUTED_KEYS
                or col in seen_chapter
            ):
                continue
            seen_chapter.add(col)
            chapter_fields.append(col)

    chapter_key = str(chapter_title or "").strip()
    parent = next(
        (
            ch
            for ch in list(chapters or [])
            if str((ch or {}).get("title", "")).strip() == chapter_key
        ),
        None,
    )
    return ffmetadata, chapter_fields, global_fields, parent


def _load_split_entries_for_chapter(
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
) -> list[dict[str, Any]]:
    """Load the split/sub-clip entries for the chapter editor.

    Looks up sub-chapter rows that belong to the given chapter and returns them
    normalized to local (chapter-relative) frame coordinates.

    Legacy format: rows have a ``parent_chapter`` column linking them to the parent.
    Canonical format: rows that fall within [ch_start, ch_end] and are not the
    parent chapter row itself are treated as sub-chapters.
    """
    archive_name = str(archive or "").strip()
    chapter_key = str(chapter_title or "").strip()
    chapter_start = int(ch_start)
    chapter_end = int(ch_end)
    chapter_frame_count = max(1, int(chapter_end) - int(chapter_start))
    if not archive_name or chapter_end <= chapter_start:
        return _default_split_entries_for_chapter(chapter_key, chapter_frame_count)

    _header, rows = _read_chapters_tsv_rows(_chapters_tsv_path(archive_name))
    if not rows:
        return _default_split_entries_for_chapter(chapter_key, chapter_frame_count)

    sub_entries: list[dict[str, Any]] = []

    # Legacy format: rows have a parent_chapter column.
    is_legacy = any(_row_ci_get(row, "parent_chapter") is not None for row in rows)

    if is_legacy:
        for row in rows:
            parent = _row_ci_get(row, "parent_chapter")
            if str(parent or "").strip() != chapter_key:
                continue
            start_frame, end_frame = _chapter_row_bounds(row)
            if start_frame is None or end_frame is None or end_frame <= start_frame:
                continue
            title = _chapter_row_title(row) or chapter_key
            sub_entries.append(
                {
                    "start_frame": int(start_frame) - chapter_start,
                    "end_frame": int(end_frame) - chapter_start,
                    "title": title,
                }
            )
    else:
        # Canonical format: find rows within [chapter_start, chapter_end] that are
        # not the parent chapter row itself.
        for row in rows:
            if _chapter_row_matches(row, chapter_key, chapter_start):
                continue
            start_frame, end_frame = _chapter_row_bounds(row)
            if start_frame is None or end_frame is None or end_frame <= start_frame:
                continue
            if int(start_frame) >= chapter_start and int(end_frame) <= chapter_end:
                title = _chapter_row_title(row) or chapter_key
                sub_entries.append(
                    {
                        "start_frame": int(start_frame) - chapter_start,
                        "end_frame": int(end_frame) - chapter_start,
                        "title": title,
                    }
                )

    if not sub_entries:
        return _default_split_entries_for_chapter(chapter_key, chapter_frame_count)

    return _normalize_split_entries_payload(
        sub_entries,
        chapter_frame_count=chapter_frame_count,
    ) or _default_split_entries_for_chapter(chapter_key, chapter_frame_count)


def _save_split_entries_for_chapter(
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
    local_entries: list[dict[str, Any]],
) -> tuple[Path, int]:
    archive_name = str(archive or "").strip()
    chapter_key = str(chapter_title or "").strip()
    chapter_start = int(ch_start)
    chapter_end = int(ch_end)
    chapter_frame_count = max(1, int(chapter_end) - int(chapter_start))
    path = _chapters_tsv_path(archive_name)
    if not archive_name:
        return path, 0
    if chapter_end <= chapter_start:
        _write_chapters_tsv_rows(path, CHAPTERS_TSV_META_COLUMNS + ["title"], [])
        return path, 0

    normalized_local = _normalize_split_entries_payload(
        local_entries,
        chapter_frame_count=chapter_frame_count,
    )
    existing_header, existing_rows = _canonical_chapters_base(path, archive_name)

    ffmetadata, chapter_fields, global_fields, ffmeta_chapter = (
        _chapters_ffmetadata_context(archive_name, chapter_key)
    )

    # Build the list of (global_start, global_end, title) entries to upsert.
    # Each is an independent chapter range — no parent-child relationship.
    entries_to_save: list[tuple[int, int, str]] = []
    if normalized_local:
        for idx, item in enumerate(list(normalized_local)):
            local_start = int(item.get("start_frame", 0))
            local_end = int(item.get("end_frame", 0))
            if local_end <= local_start:
                continue
            global_start = chapter_start + local_start
            global_end = chapter_start + local_end
            title = (
                _normalize_subtitle_optional_text(item.get("title"))
                or chapter_key
                or f"Chapter {idx + 1}"
            )
            entries_to_save.append((global_start, global_end, title))
    if not entries_to_save:
        entries_to_save.append((chapter_start, chapter_end, chapter_key))

    # Map (title, start_frame) → (global_start, global_end, title) for the entries we're saving.
    entries_key_map: dict[tuple[str, int], tuple[int, int, str]] = {
        (title, global_start): (global_start, global_end, title)
        for global_start, global_end, title in entries_to_save
    }

    # Build a template row for constructing new rows that don't yet exist in the TSV.
    # Use any existing row as the base so metadata fields (archive name, dates, etc.) are inherited.
    # Fall back to building from ffmetadata when no existing rows are available.
    template_row: dict[str, Any] = {}
    existing_template = next((dict(r) for r in existing_rows if r), None)
    if existing_template is not None:
        template_row = existing_template
    else:
        chapter_defaults: dict[str, str] = {}
        if isinstance(ffmeta_chapter, dict):
            for key in chapter_fields:
                if key in CHAPTER_FFMETADATA_COMPUTED_KEYS:
                    continue
                if key == "start":
                    chapter_defaults[key] = str(int(chapter_start))
                elif key == "end":
                    chapter_defaults[key] = str(int(chapter_end))
                elif key == "title":
                    chapter_defaults[key] = chapter_key
                else:
                    chapter_defaults[key] = str(
                        ffmeta_chapter.get(key, "") or ""
                    ).strip()
        chapter_order = [
            key for key in chapter_fields if key not in CHAPTER_FFMETADATA_COMPUTED_KEYS
        ]
        for key in global_fields:
            template_row[f"{TSV_FFMETA_PREFIX}{key}"] = str(
                ffmetadata.get(key, "") or ""
            ).strip()
        for key in chapter_order:
            template_row[key] = str(chapter_defaults.get(key, "") or "")
        if "title" not in {str(k).strip().lower() for k in chapter_order}:
            template_row["title"] = chapter_key

    # Merge: chapters are flat, independent ranges — no parent-child dropping.
    # For exact title+start matches, update the existing row in place.
    # When saving a single entry for the loaded chapter, treat it as editing that
    # loaded row in place even if its start/title changed in the editor.
    # All other existing rows are kept exactly as-is.
    single_entry_update = len(entries_to_save) == 1
    single_entry = entries_to_save[0] if single_entry_update else None
    loaded_chapter_key = (chapter_key, chapter_start)
    replaced_loaded_chapter = False
    placed: set[tuple[str, int]] = set()
    merged_rows: list[dict[str, Any]] = []
    for raw_row in list(existing_rows):
        row = dict(raw_row or {})
        start_frame, _end_frame = _chapter_row_bounds(row)
        row_title = _chapter_row_title(row)
        if start_frame is None:
            merged_rows.append(row)
            continue
        row_key = (row_title, start_frame)
        if row_key in entries_key_map:
            if row_key not in placed:
                new_start, new_end, new_title = entries_key_map[row_key]
                merged_rows.append(
                    _build_chapter_row_from_template(
                        row, existing_header, new_start, new_end, new_title
                    )
                )
                placed.add((new_title, new_start))
            # else: duplicate row for the same entry — drop it.
        elif (
            single_entry_update
            and row_key == loaded_chapter_key
            and single_entry is not None
        ):
            if not replaced_loaded_chapter:
                new_start, new_end, new_title = single_entry
                merged_rows.append(
                    _build_chapter_row_from_template(
                        row, existing_header, new_start, new_end, new_title
                    )
                )
                placed.add((new_title, new_start))
                replaced_loaded_chapter = True
            # else: duplicate row for the loaded chapter — drop it.
        else:
            merged_rows.append(row)

    # Append new rows for entries not yet present in the TSV.
    for global_start, global_end, title in entries_to_save:
        if (title, global_start) not in placed:
            merged_rows.append(
                _build_chapter_row_from_template(
                    template_row, existing_header, global_start, global_end, title
                )
            )

    merged_rows = _reindex_canonical_chapter_rows(merged_rows)
    columns: list[str] = []
    seen_columns: set[str] = set()
    seen_columns_lower: set[str] = set()

    def _add_col(raw_col: Any) -> None:
        col = str(raw_col or "").strip()
        col_lower = col.lower()
        if not col or col in seen_columns or col_lower in seen_columns_lower:
            return
        seen_columns.add(col)
        seen_columns_lower.add(col_lower)
        columns.append(col)

    for col in CHAPTERS_TSV_META_COLUMNS:
        _add_col(col)
    for col in global_fields:
        _add_col(f"{TSV_FFMETA_PREFIX}{col}")
    chapter_order_template = _chapter_order_keys_for_row(existing_header)
    for col in chapter_order_template:
        _add_col(col)
    if "title" not in {str(c).strip().lower() for c in chapter_order_template}:
        _add_col("title")
    for col in list(existing_header or []):
        _add_col(col)
    for row in merged_rows:
        for col in list((row or {}).keys()):
            _add_col(col)

    _write_chapters_tsv_rows(path, columns, merged_rows)
    return path, len(entries_to_save)


def _parse_frame_value(raw: Any) -> int | None:
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    if not re.fullmatch(r"-?\d+", text):
        return None
    try:
        value = int(text)
    except Exception:
        return None
    if value < 0:
        return None
    return int(value)


def _parse_tsv_time_or_frame_seconds(raw: Any) -> float | None:
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    # Backward compatibility: old TSV files stored archive-global frames.
    if re.fullmatch(r"-?\d+", text):
        frame = _parse_frame_value(text)
        if frame is None:
            return None
        return _frame_to_seconds(int(frame))
    return _parse_timestamp_seconds(text)


def _read_people_tsv_rows(path: Path) -> list[tuple[float, float, str]]:
    rows: list[tuple[float, float, str]] = []
    p = Path(path)
    if not p.exists():
        return rows
    for raw in p.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = str(raw or "").strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower.startswith("start_frame\t") or lower.startswith(
            "start_frame,end_frame"
        ):
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
    items = []
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
        items.append((max(0.0, float(a)), max(0.0, float(b)), text))
    if not items:
        return []
    items.sort(key=lambda item: (item[0], item[1], item[2].lower()))
    out: list[tuple[float, float, str]] = []
    for start, end, people in items:
        if out:
            prev_start, prev_end, prev_people = out[-1]
            if prev_people == people and float(prev_end) + 0.001 >= float(start):
                out[-1] = (prev_start, max(float(prev_end), float(end)), prev_people)
                continue
        out.append((round(float(start), 3), round(float(end), 3), people))
    return out


def _write_people_tsv_rows(path: Path, rows: list[tuple[float, float, str]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [PEOPLE_TSV_HEADER]
    for start, end, people in list(rows or []):
        lines.append(
            f"{_seconds_to_timestamp(float(start))}\t{_seconds_to_timestamp(float(end))}\t{str(people)}"
        )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_subtitles_tsv_rows(
    path: Path,
) -> list[tuple[float, float, str, str, float | None, str]]:
    rows: list[tuple[float, float, str, str, float | None, str]] = []
    p = Path(path)
    if not p.exists():
        return rows
    for raw in p.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = str(raw or "").strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower.startswith("start_frame\t") or lower.startswith(
            "start_frame,end_frame"
        ):
            continue
        if lower.startswith("start\t") or lower.startswith("start,end"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(",")
        if len(parts) < 3:
            continue
        start = _parse_tsv_time_or_frame_seconds(parts[0])
        end = _parse_tsv_time_or_frame_seconds(parts[1])
        text = _normalize_subtitle_optional_text(parts[2])
        speaker = _normalize_subtitle_optional_text(parts[3]) if len(parts) >= 4 else ""
        confidence = _parse_subtitle_confidence(parts[4]) if len(parts) >= 5 else None
        source = _normalize_subtitle_optional_text(parts[5]) if len(parts) >= 6 else ""
        if start is None or end is None or not text:
            continue
        if float(end) <= float(start):
            if abs(float(end) - float(start)) < 1e-9:
                end = float(start) + _frame_to_seconds(1)
            else:
                continue
        rows.append((float(start), float(end), text, speaker, confidence, source))
    return rows


def _canonicalize_subtitles_tsv_rows(
    rows: list[tuple[float, float, str, str, float | None, str]],
) -> list[tuple[float, float, str, str, float | None, str]]:
    items = []
    for start, end, text, speaker, confidence, source in list(rows or []):
        try:
            a = float(start)
            b = float(end)
        except Exception:
            continue
        if float(b) <= float(a):
            continue
        subtitle_text = _normalize_subtitle_optional_text(text)
        if not subtitle_text:
            continue
        items.append(
            (
                max(0.0, float(a)),
                max(0.0, float(b)),
                subtitle_text,
                _normalize_subtitle_optional_text(speaker),
                _parse_subtitle_confidence(confidence),
                _normalize_subtitle_optional_text(source),
            )
        )
    if not items:
        return []
    items.sort(key=lambda item: (item[0], item[1], item[2].lower(), item[3].lower()))
    return items


def _write_subtitles_tsv_rows(
    path: Path,
    rows: list[tuple[float, float, str, str, float | None, str]],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [SUBTITLES_TSV_HEADER]
    for start, end, text, speaker, confidence, source in list(rows or []):
        lines.append(
            "\t".join(
                [
                    _seconds_to_timestamp(float(start)),
                    _seconds_to_timestamp(float(end)),
                    _normalize_subtitle_optional_text(text),
                    _normalize_subtitle_optional_text(speaker),
                    _format_subtitle_confidence(confidence),
                    _normalize_subtitle_optional_text(source),
                ]
            )
        )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_people_entries_for_chapter(
    archive: str, ch_start: int, ch_end: int
) -> list[dict[str, Any]]:
    archive_name = str(archive or "").strip()
    if not archive_name:
        return []
    path = METADATA_DIR / archive_name / "people.tsv"
    if not path.exists():
        return []
    chapter_start = int(ch_start)
    chapter_end = int(ch_end)
    if chapter_end <= chapter_start:
        return []
    chapter_start_sec = _frame_to_seconds(chapter_start)
    chapter_end_sec = _frame_to_seconds(chapter_end)
    local_entries = []
    for start, end, people in _read_people_tsv_rows(path):
        lo = max(float(start), float(chapter_start_sec))
        hi = min(float(end), float(chapter_end_sec))
        if float(hi) <= float(lo):
            continue
        local_entries.append(
            {
                "start_seconds": max(0.0, float(lo) - float(chapter_start_sec)),
                "end_seconds": max(0.0, float(hi) - float(chapter_start_sec)),
                "people": people,
            }
        )
    return _normalize_people_entries_payload(
        local_entries,
        chapter_duration_seconds=max(
            0.0, float(chapter_end_sec) - float(chapter_start_sec)
        ),
    )


def _save_people_entries_for_chapter(
    archive: str,
    ch_start: int,
    ch_end: int,
    local_entries: list[dict[str, Any]],
) -> tuple[Path, int]:
    archive_name = str(archive or "").strip()
    path = METADATA_DIR / archive_name / "people.tsv"
    chapter_start = int(ch_start)
    chapter_end = int(ch_end)
    chapter_start_sec = _frame_to_seconds(chapter_start)
    chapter_end_sec = _frame_to_seconds(chapter_end)
    chapter_len_sec = max(
        _frame_to_seconds(1), float(chapter_end_sec) - float(chapter_start_sec)
    )
    if chapter_end <= chapter_start:
        _write_people_tsv_rows(
            path, _canonicalize_people_tsv_rows(_read_people_tsv_rows(path))
        )
        return path, 0

    existing = _read_people_tsv_rows(path)
    kept: list[tuple[float, float, str]] = []
    for start, end, people in existing:
        if float(end) <= float(chapter_start_sec) or float(start) >= float(
            chapter_end_sec
        ):
            kept.append((float(start), float(end), str(people)))
            continue
        if float(start) < float(chapter_start_sec):
            kept.append((float(start), float(chapter_start_sec), str(people)))
        if float(end) > float(chapter_end_sec):
            kept.append((float(chapter_end_sec), float(end), str(people)))

    normalized_local = _normalize_people_entries_payload(
        local_entries,
        chapter_duration_seconds=max(0.0, float(chapter_len_sec)),
    )
    chapter_rows: list[tuple[float, float, str]] = []
    for item in normalized_local:
        start_local = _parse_timestamp_seconds(
            item.get("start_seconds", item.get("start"))
        )
        end_local = _parse_timestamp_seconds(item.get("end_seconds", item.get("end")))
        if start_local is None or end_local is None or end_local <= start_local:
            continue
        people = re.sub(r"\s+", " ", str(item.get("people", "")).strip())
        if not people:
            continue
        local_start_sec = max(0.0, min(float(chapter_len_sec), float(start_local)))
        local_end_sec = max(0.0, min(float(chapter_len_sec), float(end_local)))
        if local_end_sec <= local_start_sec:
            if local_start_sec >= float(chapter_len_sec):
                continue
            local_end_sec = min(
                float(chapter_len_sec), float(local_start_sec) + _frame_to_seconds(1)
            )
        global_start_sec = float(chapter_start_sec) + float(local_start_sec)
        global_end_sec = float(chapter_start_sec) + float(local_end_sec)
        chapter_rows.append(
            (
                float(global_start_sec),
                float(global_end_sec),
                str(people),
            )
        )

    merged = _canonicalize_people_tsv_rows([*kept, *chapter_rows])
    _write_people_tsv_rows(path, merged)
    return path, len(chapter_rows)


def _load_subtitle_entries_for_chapter(
    archive: str, ch_start: int, ch_end: int
) -> list[dict[str, Any]]:
    archive_name = str(archive or "").strip()
    if not archive_name:
        return []
    path = METADATA_DIR / archive_name / "subtitles.tsv"
    if not path.exists():
        return []
    chapter_start = int(ch_start)
    chapter_end = int(ch_end)
    if chapter_end <= chapter_start:
        return []
    chapter_start_sec = _frame_to_seconds(chapter_start)
    chapter_end_sec = _frame_to_seconds(chapter_end)
    local_entries = []
    for start, end, text, speaker, confidence, source in _read_subtitles_tsv_rows(path):
        lo = max(float(start), float(chapter_start_sec))
        hi = min(float(end), float(chapter_end_sec))
        if float(hi) <= float(lo):
            continue
        local_entries.append(
            {
                "start_seconds": max(0.0, float(lo) - float(chapter_start_sec)),
                "end_seconds": max(0.0, float(hi) - float(chapter_start_sec)),
                "text": text,
                "speaker": speaker,
                "confidence": confidence,
                "source": source,
            }
        )
    return _normalize_subtitle_entries_payload(
        local_entries,
        chapter_duration_seconds=max(
            0.0, float(chapter_end_sec) - float(chapter_start_sec)
        ),
    )


def _save_subtitle_entries_for_chapter(
    archive: str,
    ch_start: int,
    ch_end: int,
    local_entries: list[dict[str, Any]],
) -> tuple[Path, int]:
    archive_name = str(archive or "").strip()
    path = METADATA_DIR / archive_name / "subtitles.tsv"
    chapter_start = int(ch_start)
    chapter_end = int(ch_end)
    chapter_start_sec = _frame_to_seconds(chapter_start)
    chapter_end_sec = _frame_to_seconds(chapter_end)
    chapter_len_sec = max(
        _frame_to_seconds(1), float(chapter_end_sec) - float(chapter_start_sec)
    )
    if chapter_end <= chapter_start:
        _write_subtitles_tsv_rows(
            path, _canonicalize_subtitles_tsv_rows(_read_subtitles_tsv_rows(path))
        )
        return path, 0

    existing = _read_subtitles_tsv_rows(path)
    kept: list[tuple[float, float, str, str, float | None, str]] = []
    for start, end, text, speaker, confidence, source in existing:
        if float(end) <= float(chapter_start_sec) or float(start) >= float(
            chapter_end_sec
        ):
            kept.append(
                (
                    float(start),
                    float(end),
                    str(text),
                    str(speaker),
                    confidence,
                    str(source),
                )
            )
            continue
        if float(start) < float(chapter_start_sec):
            kept.append(
                (
                    float(start),
                    float(chapter_start_sec),
                    str(text),
                    str(speaker),
                    confidence,
                    str(source),
                )
            )
        if float(end) > float(chapter_end_sec):
            kept.append(
                (
                    float(chapter_end_sec),
                    float(end),
                    str(text),
                    str(speaker),
                    confidence,
                    str(source),
                )
            )

    normalized_local = _normalize_subtitle_entries_payload(
        local_entries,
        chapter_duration_seconds=max(0.0, float(chapter_len_sec)),
    )
    chapter_rows: list[tuple[float, float, str, str, float | None, str]] = []
    for item in normalized_local:
        start_local = _parse_timestamp_seconds(
            item.get("start_seconds", item.get("start"))
        )
        end_local = _parse_timestamp_seconds(item.get("end_seconds", item.get("end")))
        if start_local is None or end_local is None or end_local <= start_local:
            continue
        text = _normalize_subtitle_optional_text(item.get("text", ""))
        if not text:
            continue
        local_start_sec = max(0.0, min(float(chapter_len_sec), float(start_local)))
        local_end_sec = max(0.0, min(float(chapter_len_sec), float(end_local)))
        if local_end_sec <= local_start_sec:
            if local_start_sec >= float(chapter_len_sec):
                continue
            local_end_sec = min(
                float(chapter_len_sec), float(local_start_sec) + _frame_to_seconds(1)
            )
        global_start_sec = float(chapter_start_sec) + float(local_start_sec)
        global_end_sec = float(chapter_start_sec) + float(local_end_sec)
        chapter_rows.append(
            (
                float(global_start_sec),
                float(global_end_sec),
                text,
                _normalize_subtitle_optional_text(item.get("speaker", "")),
                _parse_subtitle_confidence(item.get("confidence")),
                _normalize_subtitle_optional_text(item.get("source", "")),
            )
        )

    merged = _canonicalize_subtitles_tsv_rows([*kept, *chapter_rows])
    _write_subtitles_tsv_rows(path, merged)
    return path, len(chapter_rows)


def _apply_profiles_from_payload(
    session: SessionState, payload: dict[str, Any] | None
) -> None:
    payload = payload or {}
    if "force_all_frames_good" in payload:
        session.force_all_frames_good = _normalize_payload_bool(
            payload.get("force_all_frames_good"),
            default=session.force_all_frames_good,
        )

    raw_gamma_profile = payload.get("gamma_profile")
    if raw_gamma_profile is None:
        raw_gamma_profile = payload.get("gamma")
    if isinstance(raw_gamma_profile, dict):
        session.gamma_default = _normalize_gamma_value(
            raw_gamma_profile.get("default_gamma", session.gamma_default),
            default=session.gamma_default,
        )
        session.gamma_ranges = _normalize_gamma_ranges_payload(
            raw_gamma_profile.get("ranges", session.gamma_ranges),
            ch_start=session.start_frame,
            ch_end=session.end_frame,
        )

    raw_audio_sync = payload.get("audio_sync_profile")
    if isinstance(raw_audio_sync, dict):
        try:
            session.audio_sync_offset = float(
                raw_audio_sync.get("offset_seconds", session.audio_sync_offset)
            )
        except Exception:
            pass

    chapter_duration = max(
        0.0,
        _frame_to_seconds(session.end_frame) - _frame_to_seconds(session.start_frame),
    )
    raw_people_profile = payload.get("people_profile")
    if raw_people_profile is None:
        raw_people_profile = payload.get("people")
    if isinstance(raw_people_profile, dict):
        session.people_entries = _normalize_people_entries_payload(
            raw_people_profile.get("entries", session.people_entries),
            chapter_duration_seconds=chapter_duration,
        )
    elif isinstance(raw_people_profile, list):
        session.people_entries = _normalize_people_entries_payload(
            raw_people_profile,
            chapter_duration_seconds=chapter_duration,
        )

    raw_subtitles_profile = payload.get("subtitles_profile")
    if raw_subtitles_profile is None:
        raw_subtitles_profile = payload.get("subtitles")
    if isinstance(raw_subtitles_profile, dict):
        session.subtitle_entries = _normalize_subtitle_entries_payload(
            raw_subtitles_profile.get("entries", session.subtitle_entries),
            chapter_duration_seconds=chapter_duration,
        )
    elif isinstance(raw_subtitles_profile, list):
        session.subtitle_entries = _normalize_subtitle_entries_payload(
            raw_subtitles_profile,
            chapter_duration_seconds=chapter_duration,
        )

    chapter_frame_count = max(1, int(session.end_frame) - int(session.start_frame))
    raw_split_profile = payload.get("split_profile")
    if raw_split_profile is None:
        raw_split_profile = payload.get("split")
    if isinstance(raw_split_profile, dict):
        session.split_entries = _normalize_split_entries_payload(
            raw_split_profile.get("entries", session.split_entries),
            chapter_frame_count=chapter_frame_count,
        )
    elif isinstance(raw_split_profile, list):
        session.split_entries = _normalize_split_entries_payload(
            raw_split_profile,
            chapter_frame_count=chapter_frame_count,
        )


def _persist_session_progress(
    session: SessionState,
) -> tuple[Path | None, Path, Path, int, int, int, int, int]:
    out_path, count, analyzed, err = persist_bad_frames_for_chapter(
        archive=session.archive,
        chapter_title=session.chapter,
        ch_start=session.start_frame,
        ch_end=session.end_frame,
        fids=session.fids,
        sigs=session.sigs,
        overrides=session.overrides,
        wc=session.wc,
        wn=session.wn,
        wt=session.wt,
        ww=session.ww,
        tm=session.t_mode,
        ik=session.iqr_k,
        tv=session.tval,
        bp=session.bpct,
        progress=None,
        force_all_frames_good=bool(session.force_all_frames_good),
    )
    if err:
        raise RuntimeError(str(err))

    gamma_path = update_chapter_gamma_in_render_settings(
        archive=session.archive,
        ch_start=session.start_frame,
        ch_end=session.end_frame,
        gamma_ranges=session.gamma_ranges,
        default_gamma=session.gamma_default,
    )
    people_path, people_count = _save_people_entries_for_chapter(
        archive=session.archive,
        ch_start=session.start_frame,
        ch_end=session.end_frame,
        local_entries=session.people_entries,
    )
    _subtitles_path, subtitles_count = _save_subtitle_entries_for_chapter(
        archive=session.archive,
        ch_start=session.start_frame,
        ch_end=session.end_frame,
        local_entries=session.subtitle_entries,
    )
    split_path, split_count = _save_split_entries_for_chapter(
        archive=session.archive,
        chapter_title=session.chapter,
        ch_start=session.start_frame,
        ch_end=session.end_frame,
        local_entries=session.split_entries,
    )
    update_chapter_audio_sync_in_render_settings(
        archive=session.archive,
        ch_start=session.start_frame,
        ch_end=session.end_frame,
        offset_seconds=session.audio_sync_offset,
    )
    return (
        out_path,
        gamma_path,
        split_path,
        int(count),
        int(analyzed),
        int(people_count),
        int(subtitles_count),
        int(split_count),
    )


def _details_text(chapter_row: dict[str, Any] | None) -> str:
    if not chapter_row:
        return "Select a chapter."
    return (
        f"{chapter_row['title']}\n"
        f"Duration: {chapter_row['time']} | Frames: {chapter_row['frames']} | BAD already: {chapter_row['bad']}\n"
        f"Frame span: {chapter_row['start_frame']} - {chapter_row['end_frame']} (end exclusive)"
    )


def _chapter_rows_payload(
    chapters: list[dict[str, Any]], chapter_rows: list[list[Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(chapter_rows):
        ch = chapters[i] if i < len(chapters) else {}
        out.append(
            {
                "index": int(row[0]),
                "title": str(row[1]),
                "time": str(row[2]),
                "frames": int(row[3]),
                "bad": int(row[4]),
                "start_frame": int(ch.get("start_frame", 0)),
                "end_frame": int(ch.get("end_frame", 1)),
                "bad_frames": [int(x) for x in (ch.get("bad_frames", []) or [])],
            }
        )
    return out


def _archive_state(
    session: SessionState, archive: str, selected_title: str | None = None
) -> dict[str, Any]:
    data = build_archive_state(str(archive or ""), selected_title=selected_title)
    session.archive = str(archive or "")
    session.chapter = str(data["chapter_value"])
    session.chapters = list(data["chapters"])
    session.chapter_rows = list(data["chapter_rows"])
    session.start_frame = int(
        data["start_frame"] if data["start_frame"] is not None else 0
    )
    session.end_frame = int(data["end_frame"] if data["end_frame"] is not None else 1)

    rows = _chapter_rows_payload(session.chapters, session.chapter_rows)
    selected = next((r for r in rows if r["title"] == session.chapter), None)
    return {
        "archive": session.archive,
        "chapter": session.chapter,
        "status": str(data["status"]),
        "details": _details_text(selected),
        "start_frame": int(session.start_frame),
        "end_frame": int(session.end_frame),
        "chapters": rows,
    }


def _frame_status(
    session: SessionState, fid: int, score: float, thr: float
) -> tuple[str, str]:
    if bool(session.force_all_frames_good):
        return "good", "FG"
    ov = session.overrides.get(int(fid))
    if ov == "bad":
        return "bad", "MB"
    if ov == "good":
        return "good", "MG"
    if float(score) >= float(thr):
        return "bad", "AB"
    return "good", "AG"


def _frame_image_url(fid: int, *, cache_key: str = "") -> str:
    suffix = f"&rev={cache_key}" if str(cache_key or "").strip() else ""
    return f"/api/frame_image?fid={int(fid)}{suffix}"


def _frame_contact_sheet_url(
    start_index: int,
    *,
    count: int,
    columns: int,
    cache_key: str = "",
) -> str:
    suffix = f"&rev={cache_key}" if str(cache_key or "").strip() else ""
    return (
        f"/api/frame_contact_sheet?start={int(start_index)}"
        f"&count={int(count)}&columns={int(columns)}{suffix}"
    )


def _lookup_frame_image_data_url(session: SessionState, fid: int) -> str:
    fid_i = int(fid)

    def _lookup(fids: list[int], images: list[str]) -> str:
        if not fids or not images:
            return ""
        idx = bisect.bisect_left(fids, fid_i)
        if idx < len(fids) and int(fids[idx]) == fid_i and idx < len(images):
            return str(images[idx] or "")
        return ""

    return _lookup(session.fids, session.b64) or _lookup(
        session.partial_fids, session.partial_b64
    )


def _decode_frame_image_data_url(data_url: str) -> tuple[str, bytes] | None:
    text = str(data_url or "").strip()
    if not text:
        return None
    match = re.match(
        r"^data:(?P<content_type>[^;]+);base64,(?P<payload>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        payload = base64.b64decode(match.group("payload"), validate=True)
    except (binascii.Error, ValueError):
        return None
    if not payload:
        return None
    return str(match.group("content_type") or "image/jpeg"), payload


def _frame_image_cache_key(session: SessionState) -> str:
    archive = slugify(str(session.archive or "").strip()) or "archive"
    chapter = slugify(str(session.chapter or "").strip()) or "chapter"
    return f"{archive}_{chapter}_{int(session.start_frame)}_{int(session.end_frame)}"


def _contact_sheet_config_payload(session: SessionState) -> dict[str, Any]:
    return {
        "rev": _frame_image_cache_key(session),
        "chunk_size": int(CONTACT_SHEET_CHUNK_SIZE),
        "columns": int(CONTACT_SHEET_COLUMNS),
        "thumb_width": int(CONTACT_SHEET_TILE_WIDTH),
        "thumb_height": int(CONTACT_SHEET_TILE_HEIGHT),
    }


def _chapter_frame_count(session: SessionState) -> int:
    return max(0, int(session.end_frame) - int(session.start_frame))


def _contact_sheet_frame_ids(
    session: SessionState, *, start_index: int, count: int
) -> list[int]:
    total = _chapter_frame_count(session)
    if total <= 0:
        return []
    start = max(0, int(start_index))
    end = min(total, start + max(1, int(count)))
    if start >= end:
        return []
    chapter_start = int(session.start_frame)
    return [chapter_start + offset for offset in range(start, end)]


def _cached_load_signals(
    archive: str,
    chapter: str,
    *,
    video_path: Path,
    start_frame: int,
    end_frame: int,
    frame_read_offset: int,
) -> tuple:
    """Memoized wrapper around load_cached_signals — avoids re-reading the gzip file
    on every contact sheet chunk request within a single session."""
    global _SIGNALS_MEMO_KEY, _SIGNALS_MEMO_VAL
    key = (archive, chapter, str(video_path), start_frame, end_frame, frame_read_offset)
    with _SIGNALS_MEMO_LOCK:
        if _SIGNALS_MEMO_KEY == key and _SIGNALS_MEMO_VAL is not None:
            return _SIGNALS_MEMO_VAL
    result = load_cached_signals(
        archive,
        chapter,
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_read_offset=frame_read_offset,
    )
    with _SIGNALS_MEMO_LOCK:
        _SIGNALS_MEMO_KEY = key
        _SIGNALS_MEMO_VAL = result
    return result


def _load_contact_sheet_images_from_video(
    session: SessionState,
    frame_ids: list[int],
) -> dict[int, str | Image.Image]:
    video_path_raw = str(session.frame_source_video_path or "").strip()
    if not video_path_raw or not frame_ids:
        return {}
    video_path = Path(video_path_raw)
    if not video_path.exists():
        return {}

    unique_ids = sorted({int(fid) for fid in frame_ids})
    cache_fids, _cache_sigs, cached_thumbs = _cached_load_signals(
        str(session.archive or ""),
        str(session.chapter or ""),
        video_path=video_path,
        start_frame=int(session.start_frame),
        end_frame=int(session.end_frame),
        frame_read_offset=int(session.frame_source_read_offset),
    )
    _ = cache_fids
    cached_lookup = dict(cached_thumbs or {})

    out: dict[int, str | Image.Image] = {}
    missing: list[int] = []
    for fid in unique_ids:
        cached = str(cached_lookup.get(int(fid), "") or "")
        if cached:
            out[int(fid)] = cached
        else:
            missing.append(int(fid))
    if not missing:
        return out

    video_path_str = str(video_path)
    with session._video_cap_lock:
        # Reuse the session's VideoCapture if it's already open for this video.
        if (
            session._video_cap is None
            or session._video_cap_path != video_path_str
            or not session._video_cap.isOpened()
        ):
            if session._video_cap is not None:
                session._video_cap.release()
            session._video_cap = cv2.VideoCapture(video_path_str)
            session._video_cap_path = video_path_str
            session._video_cap_last_fid = -1
            if not session._video_cap.isOpened():
                session._video_cap.release()
                session._video_cap = None
                return out
        cap = session._video_cap
        read_offset = int(session.frame_source_read_offset)
        prev_read_fid = session._video_cap_last_fid
        for fid in missing:
            read_fid = int(fid) - read_offset
            if read_fid < 0:
                prev_read_fid = -1
                continue  # frame before video start — leave missing so tile is not cached
            if prev_read_fid < 0 or read_fid != prev_read_fid + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(read_fid))
            ok, read_bgr = cap.read()
            if not ok or read_bgr is None:
                prev_read_fid = -1
                continue  # read failure — leave missing so tile is not cached
            prev_read_fid = read_fid
            # Return PIL Image directly — avoids JPEG encode/decode roundtrip in the contact sheet path
            h, w = read_bgr.shape[:2]
            thumb_h = int(160 * h / max(w, 1))
            thumb_bgr = cv2.resize(
                read_bgr, (160, thumb_h), interpolation=cv2.INTER_AREA
            )
            out[int(fid)] = Image.fromarray(cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB))
        session._video_cap_last_fid = prev_read_fid
    return out


def _contact_sheet_image_data_urls(
    session: SessionState,
    *,
    start_index: int,
    count: int,
) -> list[str | Image.Image]:
    frame_ids = _contact_sheet_frame_ids(
        session,
        start_index=int(start_index),
        count=int(count),
    )
    if not frame_ids:
        return []

    image_lookup: dict[int, str | Image.Image] = {}
    missing: list[int] = []
    for fid in frame_ids:
        data_url = str(_lookup_frame_image_data_url(session, fid) or "")
        if data_url:
            image_lookup[int(fid)] = data_url
        else:
            missing.append(int(fid))
    if missing:
        image_lookup.update(_load_contact_sheet_images_from_video(session, missing))
    return [image_lookup.get(int(fid), "") or "" for fid in frame_ids]


def _fit_pil_to_contact_tile(img: Image.Image) -> Image.Image:
    """Fit a PIL Image (already decoded) into the contact sheet tile. No JPEG decode step."""
    tw, th = int(CONTACT_SHEET_TILE_WIDTH), int(CONTACT_SHEET_TILE_HEIGHT)
    if img.size == (tw, th):
        return img  # already the right size — no copy, no resize, no paste needed
    fitted = Image.new("RGB", (tw, th), (0, 0, 0))
    contained = ImageOps.contain(img, (tw, th))
    offset_x = max(0, (tw - contained.width) // 2)
    offset_y = max(0, (th - contained.height) // 2)
    fitted.paste(contained, (offset_x, offset_y))
    return fitted


def _fit_thumb_to_contact_tile(data_url: str) -> Image.Image:
    tw, th = int(CONTACT_SHEET_TILE_WIDTH), int(CONTACT_SHEET_TILE_HEIGHT)
    decoded = _decode_frame_image_data_url(data_url)
    if decoded is None:
        return Image.new("RGB", (tw, th), (0, 0, 0))
    _content_type, payload = decoded
    try:
        with Image.open(io.BytesIO(payload)) as img:
            src = (
                img.copy()
            )  # JPEG thumbnails from our encoder are always RGB — no convert needed
    except Exception:
        return Image.new("RGB", (tw, th), (0, 0, 0))
    if src.size == (tw, th):
        return src  # already the right size — skip resize and paste
    fitted = Image.new("RGB", (tw, th), (0, 0, 0))
    contained = ImageOps.contain(src, (tw, th))
    offset_x = max(0, (tw - contained.width) // 2)
    offset_y = max(0, (th - contained.height) // 2)
    fitted.paste(contained, (offset_x, offset_y))
    return fitted


def _build_contact_sheet_bytes(
    session: SessionState,
    *,
    start_index: int,
    count: int,
    columns: int,
) -> tuple[tuple[str, bytes] | None, bool]:
    """Returns (result, all_loaded). all_loaded=False means some frames failed — don't cache."""
    images = _contact_sheet_image_data_urls(
        session,
        start_index=int(start_index),
        count=int(count),
    )
    if not images:
        return None, False
    start = max(0, int(start_index))
    max_count = max(1, int(count))
    cols = max(1, int(columns))
    actual_count = max(0, min(len(images), max_count))
    if actual_count <= 0:
        return None, False
    all_loaded = all(img for img in images[:actual_count])
    rows = max(1, (actual_count + cols - 1) // cols)
    canvas = Image.new(
        "RGB",
        (int(CONTACT_SHEET_TILE_WIDTH) * cols, int(CONTACT_SHEET_TILE_HEIGHT) * rows),
        (0, 0, 0),
    )
    for local_index, item in enumerate(images[:actual_count]):
        if isinstance(item, Image.Image):
            tile = _fit_pil_to_contact_tile(item)
        else:
            tile = _fit_thumb_to_contact_tile(str(item or ""))
        x = (local_index % cols) * int(CONTACT_SHEET_TILE_WIDTH)
        y = (local_index // cols) * int(CONTACT_SHEET_TILE_HEIGHT)
        canvas.paste(tile, (x, y))

    out = io.BytesIO()
    canvas.save(out, format="JPEG", quality=72)
    return ("image/jpeg", out.getvalue()), all_loaded


def _cached_contact_sheet_bytes(
    session: SessionState,
    *,
    start_index: int,
    count: int,
    columns: int,
) -> tuple[tuple[str, bytes] | None, bool]:
    """Returns (result, all_loaded). all_loaded=True means the sheet is complete and safe to cache."""
    cache_key = (
        f"{_frame_image_cache_key(session)}|"
        f"{int(start_index)}|{int(count)}|{int(columns)}"
    )
    with _FRAME_CONTACT_SHEET_CACHE_LOCK:
        cached = _FRAME_CONTACT_SHEET_CACHE.get(cache_key)
        if cached is not None:
            _FRAME_CONTACT_SHEET_CACHE.move_to_end(cache_key)
            return cached, True

    prof_path = str(os.environ.get("VHS_PROFILE_FRAMES", "")).strip()
    if prof_path and _PROF_LOCK.acquire(blocking=False):
        try:
            _profiler = cProfile.Profile()
            built, all_loaded = _profiler.runcall(
                _build_contact_sheet_bytes,
                session,
                start_index=int(start_index),
                count=int(count),
                columns=int(columns),
            )
            _profiler.dump_stats(prof_path)
        finally:
            _PROF_LOCK.release()
    else:
        built, all_loaded = _build_contact_sheet_bytes(
            session,
            start_index=int(start_index),
            count=int(count),
            columns=int(columns),
        )
    if built is None:
        return None, False

    if all_loaded:
        with _FRAME_CONTACT_SHEET_CACHE_LOCK:
            _FRAME_CONTACT_SHEET_CACHE[cache_key] = built
            _FRAME_CONTACT_SHEET_CACHE.move_to_end(cache_key)
            while len(_FRAME_CONTACT_SHEET_CACHE) > int(CONTACT_SHEET_CACHE_LIMIT):
                _FRAME_CONTACT_SHEET_CACHE.popitem(last=False)
    return built, all_loaded


def _build_review_payload(
    session: SessionState,
    include_images: bool,
    *,
    image_url_builder: Callable[[int], str] | None = None,
) -> dict[str, Any]:
    if not session.fids or not session.sigs:
        return {
            "threshold": 0.0,
            "stats": {"total": 0, "bad": 0, "good": 0, "shown": 0, "overrides": 0},
            "force_all_frames_good": bool(session.force_all_frames_good),
            "frames": [],
        }

    scores = combined_score(
        session.sigs, session.wc, session.wn, session.wt, session.ww
    )
    thr = float(
        compute_threshold(
            scores, session.t_mode, session.iqr_k, session.tval, session.bpct
        )
    )
    session.threshold = thr

    frames: list[dict[str, Any]] = []
    bad = 0
    for i, fid in enumerate(session.fids):
        score = float(scores[i])
        status, source = _frame_status(session, int(fid), score, thr)
        if status == "bad":
            bad += 1
        frame_item: dict[str, Any] = {
            "fid": int(fid),
            "status": status,
            "source": source,
            "score": round(score, 4),
            "label": f"G:{int(fid)}  L:{max(0, int(fid) - int(session.start_frame))}  s={score:.2f}  {source}",
        }
        if include_images:
            frame_item["image"] = (
                image_url_builder(int(fid))
                if callable(image_url_builder)
                else session.b64[i]
            )
        frames.append(frame_item)

    total = len(session.fids)
    return {
        "threshold": round(thr, 4),
        "stats": {
            "total": total,
            "bad": int(bad),
            "good": int(total - bad),
            "shown": total,
            "overrides": int(len(session.overrides)),
        },
        "force_all_frames_good": bool(session.force_all_frames_good),
        "frames": frames,
    }


def _build_partial_review_payload(
    session: SessionState,
    include_images: bool,
    *,
    image_url_builder: Callable[[int], str] | None = None,
) -> dict[str, Any]:
    total = min(
        len(session.partial_fids),
        len(session.partial_b64),
        len(session.partial_sigs.get("chroma", [])),
        len(session.partial_sigs.get("noise", [])),
        len(session.partial_sigs.get("tear", [])),
        len(session.partial_sigs.get("wave", [])),
    )
    if total <= 0:
        return {
            "threshold": 0.0,
            "stats": {"total": 0, "bad": 0, "good": 0, "shown": 0, "overrides": 0},
            "force_all_frames_good": bool(session.force_all_frames_good),
            "frames": [],
        }
    tmp = SessionState()
    tmp.start_frame = int(session.start_frame)
    tmp.fids = [int(x) for x in session.partial_fids[:total]]
    tmp.b64 = list(session.partial_b64[:total])
    tmp.sigs = {
        "chroma": np.asarray(session.partial_sigs["chroma"][:total], dtype=np.float64),
        "noise": np.asarray(session.partial_sigs["noise"][:total], dtype=np.float64),
        "tear": np.asarray(session.partial_sigs["tear"][:total], dtype=np.float64),
        "wave": np.asarray(session.partial_sigs["wave"][:total], dtype=np.float64),
    }
    tmp.overrides = dict(session.overrides)
    tmp.force_all_frames_good = bool(session.force_all_frames_good)
    tmp.wc = float(session.wc)
    tmp.wn = float(session.wn)
    tmp.wt = float(session.wt)
    tmp.ww = float(session.ww)
    tmp.t_mode = str(session.t_mode)
    tmp.iqr_k = float(session.iqr_k)
    tmp.tval = float(session.tval)
    tmp.bpct = float(session.bpct)
    return _build_review_payload(
        tmp, include_images=include_images, image_url_builder=image_url_builder
    )


def _selected_bad_frame_ids(session: SessionState) -> list[int]:
    if not session.fids or not session.sigs:
        return []
    scores = combined_score(
        session.sigs, session.wc, session.wn, session.wt, session.ww
    )
    thr = float(
        compute_threshold(
            scores, session.t_mode, session.iqr_k, session.tval, session.bpct
        )
    )
    out: list[int] = []
    for fid, score in zip(session.fids, scores):
        status, _src = _frame_status(session, int(fid), float(score), thr)
        if status == "bad":
            out.append(int(fid))
    return out


def _summary_payload(session: SessionState) -> dict[str, Any]:
    review = _build_review_payload(session, include_images=False)
    bad_ids = [str(f["fid"]) for f in review["frames"] if f["status"] == "bad"]
    preview = ", ".join(bad_ids)
    gamma_ranges = _normalize_gamma_ranges_payload(
        session.gamma_ranges,
        ch_start=session.start_frame,
        ch_end=session.end_frame,
    )
    gamma_lines = []
    if gamma_ranges:
        gamma_lines.append("Gamma ranges:")
        for item in gamma_ranges:
            gamma_lines.append(
                f"- {int(item['start_frame'])}-{int(item['end_frame'])} (end exclusive): gamma {float(item['gamma']):.3f}"
            )
    else:
        gamma_lines.append("Gamma ranges: (none)")
    gamma_text = "\n".join(gamma_lines)

    people_entries = _normalize_people_entries_payload(
        session.people_entries,
        chapter_duration_seconds=max(
            0.0,
            _frame_to_seconds(session.end_frame)
            - _frame_to_seconds(session.start_frame),
        ),
    )
    people_lines = []
    if people_entries:
        people_lines.append(f"People subtitle entries: {len(people_entries)}")
        for item in people_entries:
            people_lines.append(f"- {item['start']} - {item['end']}: {item['people']}")
    else:
        people_lines.append("People subtitle entries: (none)")
    people_text = "\n".join(people_lines)

    subtitle_entries = _normalize_subtitle_entries_payload(
        session.subtitle_entries,
        chapter_duration_seconds=max(
            0.0,
            _frame_to_seconds(session.end_frame)
            - _frame_to_seconds(session.start_frame),
        ),
    )
    subtitle_lines = []
    if subtitle_entries:
        subtitle_lines.append(f"Subtitle entries: {len(subtitle_entries)}")
        for item in subtitle_entries[:25]:
            extras = []
            if str(item.get("speaker", "")).strip():
                extras.append(f"speaker={item['speaker']}")
            confidence = _format_subtitle_confidence(item.get("confidence"))
            if confidence:
                extras.append(f"confidence={confidence}")
            if str(item.get("source", "")).strip():
                extras.append(f"source={item['source']}")
            suffix = f" ({', '.join(extras)})" if extras else ""
            subtitle_lines.append(
                f"- {item['start']} - {item['end']}: {item['text']}{suffix}"
            )
        if len(subtitle_entries) > 25:
            subtitle_lines.append(f"- +{len(subtitle_entries) - 25} more")
    else:
        subtitle_lines.append("Subtitle entries: (none)")
    subtitles_text = "\n".join(subtitle_lines)

    split_entries = _normalize_split_entries_payload(
        session.split_entries,
        chapter_frame_count=max(1, int(session.end_frame) - int(session.start_frame)),
    )
    split_lines = []
    if split_entries:
        split_lines.append(f"Chapter entries: {len(split_entries)}")
        for item in split_entries[:25]:
            split_lines.append(
                f"- {int(item['start_frame'])}-{int(item['end_frame'])} (local frames): {str(item['title'])}"
            )
        if len(split_entries) > 25:
            split_lines.append(f"- +{len(split_entries) - 25} more")
    else:
        split_lines.append("Chapter entries: (none)")
    split_text = "\n".join(split_lines)

    summary_text = (
        f"Archive: {session.archive}\n"
        f"Chapter: {session.chapter}\n"
        f"Frame span: {session.start_frame} - {session.end_frame} (end exclusive)\n"
        f"Frame load mode: full chapter (all frames)\n"
        f"IQR k: {session.iqr_k:.2f}\n"
        f"Threshold: {review['threshold']:.4f}\n"
        f"Analyzed frames: {review['stats']['total']}\n"
        f"Marked bad: {review['stats']['bad']}\n"
        f"Marked good: {review['stats']['good']}\n"
        f"Manual overrides: {review['stats']['overrides']}\n"
        f"Force all frames good: {'on' if session.force_all_frames_good else 'off'}\n"
        f"Gamma default: {float(session.gamma_default):.3f}\n"
        f"{gamma_text}\n"
        f"{people_text}\n\n"
        f"{subtitles_text}\n\n"
        f"{split_text}\n\n"
        f"BAD frame IDs (loaded set):\n{preview or '(none)'}"
    )
    return {"summary": summary_text, "review": review}


def _subtitle_prompt_from_people_entries(people_entries: list[dict[str, Any]]) -> str:
    names: list[str] = []
    seen: set[str] = set()
    for item in list(people_entries or []):
        people_raw = str(item.get("people", "")).strip()
        if not people_raw:
            continue
        for part in re.split(r"\|", people_raw):
            name = _normalize_subtitle_optional_text(part)
            if not name:
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
            if len(names) >= 25:
                break
        if len(names) >= 25:
            break
    if not names:
        return ""
    return (
        "Transcribe in English. Use these exact spellings when heard: "
        + ", ".join(names)
        + "."
    )


def _load_whisper_model() -> Any:
    global _WHISPER_MODEL
    if whisper is None:
        raise RuntimeError(
            "Whisper is unavailable. Install whisper to generate subtitles."
        )
    with _WHISPER_MODEL_LOCK:
        if _WHISPER_MODEL is None:
            _WHISPER_MODEL = whisper.load_model(
                WHISPER_MODEL_NAME, download_root=str(WHISPER_MODEL_DIR)
            )
        return _WHISPER_MODEL


def _load_whisper_transcribe_module() -> Any:
    global _WHISPER_TRANSCRIBE_MODULE
    if _WHISPER_TRANSCRIBE_MODULE is None:
        _WHISPER_TRANSCRIBE_MODULE = importlib.import_module("whisper.transcribe")
    return _WHISPER_TRANSCRIBE_MODULE


class WizardHandler(BaseHTTPRequestHandler):
    server_version = "VHSTuner/1.0"

    def log_message(self, _format: str, *args: Any) -> None:
        pass  # suppress per-request stderr noise

    def handle_error(self, request: Any, client_address: Any) -> None:
        # Silence expected client-disconnect errors (aborted/reset connections)
        import sys

        exc = sys.exc_info()[1]
        if isinstance(
            exc, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)
        ):
            return
        super().handle_error(request, client_address)

    def _ensure_session(self) -> SessionState:
        self._set_cookie: str | None = None
        cookies = SimpleCookie(self.headers.get("Cookie", ""))
        sid = cookies.get(SESSION_COOKIE)
        sid_val = sid.value if sid else ""

        with _SESSION_LOCK:
            if sid_val and sid_val in _SESSIONS:
                return _SESSIONS[sid_val]

            sid_val = uuid.uuid4().hex
            sess = SessionState()
            _SESSIONS[sid_val] = sess
            self._set_cookie = (
                f"{SESSION_COOKIE}={sid_val}; Path=/; HttpOnly; SameSite=Lax"
            )
            return sess

    def _send_json(self, payload: dict[str, Any], code: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if self._set_cookie:
            self.send_header("Set-Cookie", self._set_cookie)
        self.end_headers()
        self.wfile.write(data)

    def _send_text(
        self,
        text: str,
        code: int = HTTPStatus.OK,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        data = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if self._set_cookie:
            self.send_header("Set-Cookie", self._set_cookie)
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(
        self,
        data: bytes,
        *,
        content_type: str,
        code: int = HTTPStatus.OK,
        cache_control: str = "no-store",
    ) -> None:
        raw = bytes(data or b"")
        self.send_response(code)
        self.send_header("Content-Type", str(content_type))
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", str(cache_control))
        if self._set_cookie:
            self.send_header("Set-Cookie", self._set_cookie)
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_media_file(
        self,
        path: Path,
        *,
        content_type: str,
        missing_message: str,
        empty_message: str,
    ) -> None:
        p = Path(path)
        if not p.exists() or not p.is_file():
            self._send_error_json(str(missing_message), code=HTTPStatus.NOT_FOUND)
            return

        total_size = int(p.stat().st_size)
        if total_size <= 0:
            self._send_error_json(str(empty_message), code=HTTPStatus.NOT_FOUND)
            return

        start = 0
        end = total_size - 1
        status = HTTPStatus.OK
        content_range = None

        range_header = str(self.headers.get("Range", "") or "").strip()
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)$", range_header)
            if m:
                g_start, g_end = m.groups()
                if g_start:
                    start = int(g_start)
                if g_end:
                    end = int(g_end)
                if not g_end:
                    end = total_size - 1
                if g_start and not g_end:
                    end = total_size - 1
                if start < 0:
                    start = 0
                if end >= total_size:
                    end = total_size - 1
                if start > end or start >= total_size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{total_size}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Cache-Control", "no-store")
                    if self._set_cookie:
                        self.send_header("Set-Cookie", self._set_cookie)
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
                content_range = f"bytes {start}-{end}/{total_size}"

        length = (end - start) + 1
        self.send_response(status)
        self.send_header("Content-Type", str(content_type))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        if content_range:
            self.send_header("Content-Range", content_range)
        if self._set_cookie:
            self.send_header("Set-Cookie", self._set_cookie)
        self.end_headers()

        with p.open("rb") as fh:
            fh.seek(start)
            remaining = length
            chunk_size = 64 * 1024
            while remaining > 0:
                chunk = fh.read(min(chunk_size, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # Client disconnected mid-stream (common when media element
                    # seeks/cancels range requests). Treat as non-fatal.
                    return
                remaining -= len(chunk)

    def _send_video_file(self, path: Path) -> None:
        self._send_media_file(
            path,
            content_type="video/mp4",
            missing_message="Preview video is not available.",
            empty_message="Preview video is empty.",
        )

    def _send_audio_file(self, path: Path) -> None:
        self._send_media_file(
            path,
            content_type="audio/wav",
            missing_message="Chapter audio is not available.",
            empty_message="Chapter audio is empty.",
        )

    def _chapter_audio_cache_key(self, session: SessionState) -> str:
        return (
            f"{str(session.archive or '').strip()}|"
            f"{str(session.chapter or '').strip()}|"
            f"{int(session.start_frame)}|{int(session.end_frame)}"
        )

    def _ensure_chapter_audio_file(
        self, session: SessionState
    ) -> tuple[Path | None, str]:
        if (
            not str(session.archive or "").strip()
            or not str(session.chapter or "").strip()
        ):
            return None, "Load a chapter before requesting audio."
        if int(session.end_frame) <= int(session.start_frame):
            return None, "Invalid chapter frame span for audio."

        cache_key = self._chapter_audio_cache_key(session)
        existing_raw = str(session.chapter_audio_path or "").strip()
        existing = Path(existing_raw) if existing_raw else None
        if (
            existing
            and session.chapter_audio_key == cache_key
            and existing.exists()
            and existing.is_file()
            and int(existing.stat().st_size) > 44
        ):
            return existing, ""

        source_video = _resolve_archive_video(session.archive)
        if not source_video:
            return None, f"No archive video found for '{session.archive}'."

        start_sec = _frame_to_seconds(session.start_frame)
        end_sec = _frame_to_seconds(session.end_frame)
        if float(end_sec) <= float(start_sec):
            return None, "Invalid chapter time range for audio."

        out_dir = Path(tempfile.gettempdir()) / "vhs_plain_wizard_audio"
        out_dir.mkdir(parents=True, exist_ok=True)

        archive_slug = slugify(str(session.archive or "archive")) or "archive"
        chapter_slug = slugify(str(session.chapter or "chapter")) or "chapter"
        key_hash = uuid.uuid5(uuid.NAMESPACE_URL, cache_key).hex[:16]
        out_name = (
            f"{archive_slug}_{chapter_slug}_"
            f"{int(session.start_frame)}_{int(session.end_frame)}_{key_hash}.wav"
        )
        out_path = out_dir / out_name

        needs_extract = True
        try:
            needs_extract = (not out_path.exists()) or (
                int(out_path.stat().st_size) <= 44
            )
        except Exception:
            needs_extract = True

        if needs_extract:
            cmd = [
                str(FFMPEG_BIN),
                "-nostdin",
                "-v",
                "error",
                "-ss",
                f"{float(start_sec):.3f}",
                "-to",
                f"{float(end_sec):.3f}",
                "-i",
                str(source_video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(out_path),
            ]
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 or not out_path.exists():
                detail = (proc.stderr or proc.stdout or "").strip()
                return None, detail or "ffmpeg audio extraction failed."

        session.chapter_audio_key = cache_key
        session.chapter_audio_path = str(out_path)
        return out_path, ""

    _AUDIO_SYNC_PAD_SECONDS = 5.0

    def _audio_sync_cache_key(self, session: SessionState) -> str:
        return (
            f"sync|{str(session.archive or '').strip()}|"
            f"{str(session.chapter or '').strip()}|"
            f"{int(session.start_frame)}|{int(session.end_frame)}"
        )

    def _ensure_audio_sync_file(
        self, session: SessionState
    ) -> tuple[Path | None, str, float, float]:
        """Extract audio with ±AUDIO_SYNC_PAD_SECONDS buffer around the chapter.

        Returns (path, error_msg, padded_start_sec, chapter_start_sec).
        padded_start_sec is the actual start of the audio file (= chapter_start - pad, clamped to 0).
        chapter_start_sec is where within the audio file the chapter video begins.
        """
        if (
            not str(session.archive or "").strip()
            or not str(session.chapter or "").strip()
        ):
            return None, "Load a chapter before requesting audio sync.", 0.0, 0.0
        if int(session.end_frame) <= int(session.start_frame):
            return None, "Invalid chapter frame span for audio sync.", 0.0, 0.0

        cache_key = self._audio_sync_cache_key(session)
        existing_raw = str(session.audio_sync_audio_path or "").strip()
        existing = Path(existing_raw) if existing_raw else None
        pad = self._AUDIO_SYNC_PAD_SECONDS
        chapter_start_sec = _frame_to_seconds(session.start_frame)
        chapter_end_sec = _frame_to_seconds(session.end_frame)
        padded_start = max(0.0, float(chapter_start_sec) - pad)
        padded_end = float(chapter_end_sec) + pad
        video_offset = (
            float(chapter_start_sec) - padded_start
        )  # seconds into audio file where chapter starts

        if (
            existing
            and session.audio_sync_audio_key == cache_key
            and existing.exists()
            and existing.is_file()
            and int(existing.stat().st_size) > 44
        ):
            return existing, "", padded_start, video_offset

        source_video = _resolve_archive_video(session.archive)
        if not source_video:
            return None, f"No archive video found for '{session.archive}'.", 0.0, 0.0

        out_dir = Path(tempfile.gettempdir()) / "vhs_plain_wizard_audio"
        out_dir.mkdir(parents=True, exist_ok=True)

        archive_slug = slugify(str(session.archive or "archive")) or "archive"
        chapter_slug = slugify(str(session.chapter or "chapter")) or "chapter"
        key_hash = uuid.uuid5(uuid.NAMESPACE_URL, cache_key).hex[:16]
        out_name = (
            f"sync_{archive_slug}_{chapter_slug}_"
            f"{int(session.start_frame)}_{int(session.end_frame)}_{key_hash}.wav"
        )
        out_path = out_dir / out_name

        needs_extract = True
        try:
            needs_extract = (not out_path.exists()) or (
                int(out_path.stat().st_size) <= 44
            )
        except Exception:
            needs_extract = True

        if needs_extract:
            cmd = [
                str(FFMPEG_BIN),
                "-nostdin",
                "-v",
                "error",
                "-ss",
                f"{float(padded_start):.3f}",
                "-to",
                f"{float(padded_end):.3f}",
                "-i",
                str(source_video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(out_path),
            ]
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0 or not out_path.exists():
                detail = (proc.stderr or proc.stdout or "").strip()
                return None, detail or "ffmpeg audio sync extraction failed.", 0.0, 0.0

        session.audio_sync_audio_key = cache_key
        session.audio_sync_audio_path = str(out_path)
        return out_path, "", padded_start, video_offset

    def _preview_page_html(self, session: SessionState) -> str:
        title_text = html.escape(str(session.chapter or "Preview"))
        return (
            "<!doctype html>\n"
            '<html><head><meta charset="utf-8"><title>VHS Preview</title>'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<style>"
            "html,body{width:100%;height:100%;overflow:hidden;}"
            "body{margin:0;background:#111;color:#eee;font-family:Segoe UI,Arial,sans-serif;}"
            ".wrap{display:grid;grid-template-rows:auto minmax(0,1fr);height:100vh;gap:8px;padding:10px;box-sizing:border-box;}"
            ".meta{font-size:13px;opacity:.9;}"
            "video{display:block;width:100%;height:100%;max-width:100%;max-height:100%;box-sizing:border-box;object-fit:contain;background:#000;border:1px solid #333;border-radius:8px;}"
            '</style></head><body><div class="wrap">'
            f'<div class="meta">Preview: {title_text}</div>'
            '<video controls autoplay preload="auto" src="/api/preview_video"></video>'
            "</div></body></html>"
        )

    def _read_json(self) -> dict[str, Any]:
        raw_len = int(self.headers.get("Content-Length", "0") or "0")
        if raw_len <= 0:
            return {}
        raw = self.rfile.read(raw_len)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_error_json(
        self, message: str, code: int = HTTPStatus.BAD_REQUEST
    ) -> None:
        self._send_json({"ok": False, "error": message}, code=code)

    def do_GET(self) -> None:
        session = self._ensure_session()
        parsed = urlparse(self.path)

        if parsed.path == "/":
            if not INDEX_HTML.exists():
                self._send_text(
                    "Missing index.html", code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                return
            self._send_text(
                INDEX_HTML.read_text(encoding="utf-8"),
                content_type="text/html; charset=utf-8",
            )
            return

        if parsed.path.endswith((".js", ".css")) and "/" not in parsed.path.lstrip("/"):
            cache_key = parsed.path.lstrip("/")
            cached = _STATIC_FILE_CACHE.get(cache_key)
            if cached is None:
                static_path = STATIC_DIR / cache_key
                if not static_path.exists() or not static_path.is_file():
                    self._send_text("Not found", code=HTTPStatus.NOT_FOUND)
                    return
                content_type = (
                    "application/javascript; charset=utf-8"
                    if parsed.path.endswith(".js")
                    else "text/css; charset=utf-8"
                )
                cached = (content_type, static_path.read_bytes())
                _STATIC_FILE_CACHE[cache_key] = cached
            self._send_bytes(
                cached[1], content_type=cached[0], cache_control="no-store"
            )
            return

        if parsed.path == "/preview":
            preview_raw = str(session.preview_video_path or "").strip()
            preview_path = Path(preview_raw) if preview_raw else None
            if (
                not preview_path
                or not preview_path.exists()
                or not preview_path.is_file()
            ):
                self._send_text(
                    "Preview render is not ready yet. Run Preview Render from Step 2 first.",
                    code=HTTPStatus.NOT_FOUND,
                )
                return
            self._send_text(
                self._preview_page_html(session),
                content_type="text/html; charset=utf-8",
            )
            return

        if parsed.path == "/api/preview_video":
            preview_raw = str(session.preview_video_path or "").strip()
            preview_path = Path(preview_raw) if preview_raw else None
            if (
                not preview_path
                or not preview_path.exists()
                or not preview_path.is_file()
            ):
                self._send_error_json(
                    "Preview render is not ready yet.", code=HTTPStatus.NOT_FOUND
                )
                return
            self._send_video_file(preview_path)
            return

        if parsed.path == "/api/chapter_audio":
            audio_path, err = self._ensure_chapter_audio_file(session)
            if err or audio_path is None:
                self._send_error_json(
                    err or "Chapter audio is not available.", code=HTTPStatus.NOT_FOUND
                )
                return
            self._send_audio_file(audio_path)
            return

        if parsed.path == "/api/audio_sync_info":
            sync_path, err, padded_start, video_offset = self._ensure_audio_sync_file(
                session
            )
            if err or sync_path is None:
                self._send_error_json(
                    err or "Audio sync file is not available.",
                    code=HTTPStatus.NOT_FOUND,
                )
                return
            chapter_duration = _frame_to_seconds(session.end_frame) - _frame_to_seconds(
                session.start_frame
            )
            self._send_json(
                {
                    "ok": True,
                    "padded_start_sec": float(padded_start),
                    "video_offset_sec": float(video_offset),
                    "chapter_duration_sec": float(chapter_duration),
                    "pad_seconds": float(self._AUDIO_SYNC_PAD_SECONDS),
                    "offset_seconds": float(session.audio_sync_offset),
                }
            )
            return

        if parsed.path == "/api/audio_sync_audio":
            sync_path, err, _padded_start, _video_offset = self._ensure_audio_sync_file(
                session
            )
            if err or sync_path is None:
                self._send_error_json(
                    err or "Audio sync file is not available.",
                    code=HTTPStatus.NOT_FOUND,
                )
                return
            self._send_audio_file(sync_path)
            return

        if parsed.path == "/api/frame_image":
            params = parse_qs(parsed.query)
            raw_fid = str((params.get("fid", [""])[0] or "").strip())
            if not raw_fid or not re.fullmatch(r"-?\d+", raw_fid):
                self._send_error_json(
                    "Missing or invalid frame id.", code=HTTPStatus.BAD_REQUEST
                )
                return
            data_url = _lookup_frame_image_data_url(session, int(raw_fid))
            decoded = _decode_frame_image_data_url(data_url)
            if decoded is None:
                # Thumbnail not in session cache — fetch on-demand from video
                fid_images = _load_contact_sheet_images_from_video(
                    session, [int(raw_fid)]
                )
                fid_result = fid_images.get(int(raw_fid))
                if isinstance(fid_result, Image.Image):
                    out_buf = io.BytesIO()
                    fid_result.save(out_buf, format="JPEG", quality=85)
                    decoded = ("image/jpeg", out_buf.getvalue())
                elif isinstance(fid_result, str):
                    decoded = _decode_frame_image_data_url(fid_result)
            if decoded is None:
                self._send_error_json(
                    "Frame image is not available.", code=HTTPStatus.NOT_FOUND
                )
                return
            content_type, payload = decoded
            self._send_bytes(
                payload,
                content_type=content_type,
                cache_control="private, max-age=3600, immutable",
            )
            return

        if parsed.path == "/api/frame_contact_sheet":
            params = parse_qs(parsed.query)
            raw_start = str((params.get("start", [""])[0] or "").strip())
            raw_count = str((params.get("count", [""])[0] or "").strip())
            raw_columns = str((params.get("columns", [""])[0] or "").strip())
            if (
                not raw_start
                or not raw_count
                or not re.fullmatch(r"-?\d+", raw_start)
                or not re.fullmatch(r"-?\d+", raw_count)
                or (raw_columns and not re.fullmatch(r"-?\d+", raw_columns))
            ):
                self._send_error_json(
                    "Missing or invalid contact sheet parameters.",
                    code=HTTPStatus.BAD_REQUEST,
                )
                return
            start_index = max(0, int(raw_start))
            count = max(1, int(raw_count))
            columns = max(1, int(raw_columns or CONTACT_SHEET_COLUMNS))
            built, sheet_complete = _cached_contact_sheet_bytes(
                session,
                start_index=start_index,
                count=count,
                columns=columns,
            )
            if built is None:
                self._send_error_json(
                    "Frame contact sheet is not available.", code=HTTPStatus.NOT_FOUND
                )
                return
            content_type, payload = built
            self._send_bytes(
                payload,
                content_type=content_type,
                cache_control=(
                    "private, max-age=3600, immutable" if sheet_complete else "no-store"
                ),
            )
            return

        if parsed.path == "/api/archives":
            archives = _get_archives()
            selected = (
                session.archive
                if session.archive in archives
                else (archives[0] if archives else "")
            )
            self._send_json({"ok": True, "archives": archives, "selected": selected})
            return

        if parsed.path == "/api/archive_state":
            params = parse_qs(parsed.query)
            archive = str((params.get("archive", [""])[0] or "").strip())
            chapter = str((params.get("chapter", [""])[0] or "").strip())
            if not archive:
                archives = _get_archives()
                archive = archives[0] if archives else ""
            state = _archive_state(session, archive, selected_title=(chapter or None))
            self._send_json({"ok": True, "archive_state": state})
            return

        if parsed.path == "/api/summary":
            if not session.fids:
                self._send_error_json("No loaded chapter data yet.")
                return
            self._send_json({"ok": True, **_summary_payload(session)})
            return

        if parsed.path == "/api/load_progress":
            progress_payload: dict[str, Any] = {
                "ok": True,
                "running": bool(session.load_running),
                "progress": float(session.load_progress),
                "message": str(session.load_message or ""),
                "sample_done": int(session.load_sample_done),
                "sample_total": int(session.load_sample_total),
            }
            if session.load_meta_ready:
                progress_payload["people_profile"] = {
                    "entries": list(session.people_entries),
                    "source": "people_tsv",
                }
                progress_payload["subtitles_profile"] = {
                    "entries": list(session.subtitle_entries),
                    "source": "subtitles_tsv",
                }
            self._send_json(progress_payload)
            return

        if parsed.path == "/api/preview_progress":
            self._send_json(
                {
                    "ok": True,
                    "running": bool(session.preview_running),
                    "progress": float(session.preview_progress),
                    "message": str(session.preview_message or ""),
                    "frame_done": int(session.preview_frame_done),
                    "frame_total": int(session.preview_frame_total),
                }
            )
            return

        if parsed.path == "/api/subtitles_progress":
            self._send_json(
                {
                    "ok": True,
                    "running": bool(session.subtitles_running),
                    "progress": float(session.subtitles_progress),
                    "message": str(session.subtitles_message or ""),
                    "segment_done": int(session.subtitles_segment_done),
                    "segment_total": int(session.subtitles_segment_total),
                }
            )
            return

        if parsed.path == "/api/load_review":
            self._send_json(
                {
                    "ok": True,
                    "running": bool(session.load_running),
                    "contact_sheet": _contact_sheet_config_payload(session),
                    "review": _build_partial_review_payload(
                        session,
                        include_images=False,
                    ),
                }
            )
            return

        # -----------------------------------------------------------------------
        # AI-agent endpoints  (/api/ai/*)
        # -----------------------------------------------------------------------

        if parsed.path == "/api/ai/signal_data":
            if not session.fids or not session.sigs:
                self._send_error_json("No loaded chapter data yet.")
                return
            scores = combined_score(
                session.sigs, session.wc, session.wn, session.wt, session.ww
            )
            thr = float(
                compute_threshold(
                    scores, session.t_mode, session.iqr_k, session.tval, session.bpct
                )
            )
            bad = sum(
                1
                for i, fid in enumerate(session.fids)
                if _frame_status(session, int(fid), float(scores[i]), thr)[0] == "bad"
            )
            self._send_json(
                {
                    "ok": True,
                    "archive": str(session.archive),
                    "chapter": str(session.chapter),
                    "start_frame": int(session.start_frame),
                    "end_frame": int(session.end_frame),
                    "iqr_k": float(session.iqr_k),
                    "threshold": round(thr, 4),
                    "fids": [int(f) for f in session.fids],
                    "scores": [round(float(s), 4) for s in scores],
                    "signals": {
                        k: [round(float(x), 4) for x in v]
                        for k, v in session.sigs.items()
                    },
                    "stats": {
                        "total": len(session.fids),
                        "bad": bad,
                        "good": len(session.fids) - bad,
                        "overrides": len(session.overrides),
                    },
                }
            )
            return

        if parsed.path == "/api/ai/suggest_k":
            if not session.fids or not session.sigs:
                self._send_error_json("No loaded chapter data yet.")
                return
            scores = combined_score(
                session.sigs, session.wc, session.wn, session.wt, session.ww
            )
            current_thr = float(
                compute_threshold(
                    scores, session.t_mode, session.iqr_k, session.tval, session.bpct
                )
            )
            current_bad = sum(
                1
                for i, fid in enumerate(session.fids)
                if _frame_status(session, int(fid), float(scores[i]), current_thr)[0]
                == "bad"
            )
            result = suggest_iqr_k(scores)  # type: ignore[arg-type]
            result.update(
                {
                    "ok": True,
                    "current_k": float(session.iqr_k),
                    "current_threshold": round(current_thr, 4),
                    "current_bad_count": current_bad,
                }
            )
            self._send_json(result)
            return

        if parsed.path == "/api/ai/spike_regions":
            if not session.fids or not session.sigs:
                self._send_error_json("No loaded chapter data yet.")
                return
            params = parse_qs(parsed.query)
            context = int((params.get("context", ["8"])[0] or "8"))
            scores = combined_score(
                session.sigs, session.wc, session.wn, session.wt, session.ww
            )
            thr = float(
                compute_threshold(
                    scores, session.t_mode, session.iqr_k, session.tval, session.bpct
                )
            )
            regions = find_spike_regions(session.fids, scores, thr, context_frames=context)  # type: ignore[arg-type]
            total_bad = sum(r["bad_frame_count"] for r in regions)
            self._send_json(
                {
                    "ok": True,
                    "threshold": round(thr, 4),
                    "iqr_k": float(session.iqr_k),
                    "total_bad_frames": total_bad,
                    "total_regions": len(regions),
                    "regions": regions,
                }
            )
            return

        if parsed.path == "/api/ai/frames_in_region":
            if not session.fids or not session.sigs:
                self._send_error_json("No loaded chapter data yet.")
                return
            params = parse_qs(parsed.query)
            try:
                start_fid = int(params.get("start_fid", ["0"])[0])
                end_fid = int(params.get("end_fid", ["0"])[0])
            except (ValueError, IndexError):
                self._send_error_json("start_fid and end_fid are required integers.")
                return
            scores = combined_score(
                session.sigs, session.wc, session.wn, session.wt, session.ww
            )
            thr = float(
                compute_threshold(
                    scores, session.t_mode, session.iqr_k, session.tval, session.bpct
                )
            )
            # Find index range within session.fids
            idx_start = bisect.bisect_left(session.fids, start_fid)
            idx_end = bisect.bisect_right(session.fids, end_fid) - 1
            if idx_start > idx_end or idx_start >= len(session.fids):
                self._send_json({"ok": True, "frames": [], "contact_sheet_url": ""})
                return
            frames_out = []
            for i in range(idx_start, idx_end + 1):
                fid = int(session.fids[i])
                score = float(scores[i])
                status, source = _frame_status(session, fid, score, thr)
                sigs_at = {
                    k: round(float(v[i]), 4)
                    for k, v in session.sigs.items()
                    if i < len(v)
                }
                frames_out.append(
                    {
                        "fid": fid,
                        "local_frame": max(0, fid - int(session.start_frame)),
                        "score": round(score, 4),
                        "status": status,
                        "source": source,
                        "image_url": _frame_image_url(fid),
                        "signals": sigs_at,
                    }
                )
            count = idx_end - idx_start + 1
            contact_url = _frame_contact_sheet_url(idx_start, count=count, columns=8)
            self._send_json(
                {
                    "ok": True,
                    "frames": frames_out,
                    "contact_sheet_url": contact_url,
                }
            )
            return

        if parsed.path == "/api/ai/chapter_state":
            if not session.fids:
                self._send_error_json("No loaded chapter data yet.")
                return
            scores = combined_score(
                session.sigs, session.wc, session.wn, session.wt, session.ww
            )
            thr = float(
                compute_threshold(
                    scores, session.t_mode, session.iqr_k, session.tval, session.bpct
                )
            )
            bad = sum(
                1
                for i, fid in enumerate(session.fids)
                if _frame_status(session, int(fid), float(scores[i]), thr)[0] == "bad"
            )
            # Build boundary frame URLs: first, last, first+5, last-5
            fids = session.fids
            n = len(fids)
            boundary_urls = {
                "first": _frame_image_url(int(fids[0])),
                "first_plus_5": _frame_image_url(int(fids[min(5, n - 1)])),
                "last_minus_5": _frame_image_url(int(fids[max(0, n - 6)])),
                "last": _frame_image_url(int(fids[-1])),
            }
            self._send_json(
                {
                    "ok": True,
                    "archive": str(session.archive),
                    "chapter": str(session.chapter),
                    "start_frame": int(session.start_frame),
                    "end_frame": int(session.end_frame),
                    "duration_frames": int(session.end_frame)
                    - int(session.start_frame),
                    "iqr_k": float(session.iqr_k),
                    "threshold": round(thr, 4),
                    "bad_frame_count": bad,
                    "force_all_frames_good": bool(session.force_all_frames_good),
                    "gamma_default": float(session.gamma_default),
                    "gamma_ranges": list(session.gamma_ranges),
                    "people_entries": list(session.people_entries),
                    "subtitle_entries": list(session.subtitle_entries),
                    "split_entries": list(session.split_entries),
                    "auto_transcript": str(session.auto_transcript),
                    "boundary_frame_urls": boundary_urls,
                }
            )
            return

        if parsed.path == "/api/ai/suggest_gamma":
            if not session.fids or not session.b64:
                self._send_error_json("No loaded chapter data yet.")
                return
            # Sample ~32 frames from the middle third of the chapter
            n = len(session.b64)
            lo = n // 3
            hi = 2 * n // 3
            mid_b64 = session.b64[lo:hi]
            step = max(1, len(mid_b64) // 32)
            sample = mid_b64[::step][:32]
            result = estimate_gamma_from_frames(sample)
            result.update(
                {
                    "ok": True,
                    "current_gamma_default": float(session.gamma_default),
                }
            )
            self._send_json(result)
            return

        self._send_error_json("Not found", code=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        session = self._ensure_session()
        parsed = urlparse(self.path)

        try:
            payload = self._read_json()
        except Exception:
            self._send_error_json("Invalid JSON body")
            return

        if parsed.path == "/api/load_chapter":
            self._handle_load_chapter(session, payload)
            return

        if parsed.path == "/api/cancel_load":
            session.load_cancel_requested = True
            if session.load_running:
                _set_load_progress(
                    session,
                    message="Cancel requested... stopping after current frame.",
                )
            self._send_json(
                {
                    "ok": True,
                    "running": bool(session.load_running),
                    "message": (
                        "Cancel requested... stopping after current frame."
                        if session.load_running
                        else "No active load to cancel."
                    ),
                }
            )
            return

        if parsed.path == "/api/cancel_subtitles":
            session.subtitles_cancel_requested = True
            if session.subtitles_running:
                _set_subtitles_progress(
                    session,
                    message="Cancel requested... stopping after current decode chunk.",
                )
            self._send_json(
                {
                    "ok": True,
                    "running": bool(session.subtitles_running),
                    "message": (
                        "Cancel requested... stopping after current decode chunk."
                        if session.subtitles_running
                        else "No active subtitle generation to cancel."
                    ),
                }
            )
            return

        if parsed.path == "/api/apply_iqr":
            if not session.fids:
                self._send_error_json("No loaded chapter data yet.")
                return
            session.iqr_k = _normalize_iqr_k(
                payload.get("iqr_k", session.iqr_k), default=session.iqr_k
            )
            if "force_all_frames_good" in payload:
                session.force_all_frames_good = _normalize_payload_bool(
                    payload.get("force_all_frames_good"),
                    default=session.force_all_frames_good,
                )
            review = _build_review_payload(session, include_images=False)
            self._send_json({"ok": True, "review": review})
            return

        if parsed.path == "/api/set_force_all_good":
            self._handle_set_force_all_good(session, payload)
            return

        if parsed.path == "/api/toggle_frame":
            if not session.fids and not session.partial_fids:
                self._send_error_json("No loaded chapter data yet.")
                return
            try:
                fid = int(payload.get("fid"))
            except Exception:
                self._send_error_json("Missing or invalid frame id.")
                return
            self._handle_toggle_frame(session, fid)
            return

        if parsed.path == "/api/set_frame_range":
            if not session.fids and not session.partial_fids:
                self._send_error_json("No loaded chapter data yet.")
                return
            try:
                start_fid = int(payload.get("start_fid"))
                end_fid = int(payload.get("end_fid"))
            except Exception:
                self._send_error_json("Missing or invalid range frame ids.")
                return
            status_raw = str(payload.get("status", "bad") or "bad").strip().lower()
            status = "good" if status_raw == "good" else "bad"
            self._handle_set_frame_range(session, start_fid, end_fid, status)
            return

        if parsed.path == "/api/preview_render":
            self._handle_preview_render(session, payload)
            return

        if parsed.path == "/api/people_prefill_cast":
            self._handle_people_prefill_cast(session, payload)
            return

        if parsed.path == "/api/subtitles_generate":
            self._handle_subtitles_generate(session, payload)
            return

        if parsed.path == "/api/set_auto_transcript":
            self._handle_set_auto_transcript(session, payload)
            return

        if parsed.path == "/api/save":
            self._handle_save(session, payload)
            return

        if parsed.path == "/api/save_progress":
            self._handle_save_progress(session, payload)
            return

        if parsed.path == "/api/rename_chapter":
            archive = str(payload.get("archive", "")).strip()
            old_title = str(payload.get("old_title", "")).strip()
            new_title = str(payload.get("new_title", "")).strip()
            if not archive or not old_title or not new_title:
                self._send_error_json("Missing archive, old_title, or new_title.")
                return
            if old_title == new_title:
                self._send_json(
                    {
                        "ok": True,
                        "old_title": old_title,
                        "new_title": new_title,
                        "renamed_files": [],
                    }
                )
                return
            tsv_path = _chapters_tsv_path(archive)
            if not tsv_path.exists():
                self._send_error_json(f"No chapters.tsv found for archive: {archive}")
                return
            header, rows = _read_chapters_tsv_rows(tsv_path)
            title_key = next((k for k in header if k.strip().lower() == "title"), None)
            if not title_key:
                self._send_error_json("chapters.tsv has no 'title' column.")
                return
            matched = False
            for row in rows:
                if row.get(title_key, "").strip() == old_title:
                    row[title_key] = new_title
                    matched = True
                    break
            if not matched:
                self._send_error_json(f"Chapter not found: {old_title!r}")
                return
            _write_chapters_tsv_rows(tsv_path, header, rows)
            renamed_files = _rename_chapter_outputs(old_title, new_title)
            self._send_json(
                {
                    "ok": True,
                    "old_title": old_title,
                    "new_title": new_title,
                    "renamed_files": renamed_files,
                }
            )
            return

        if parsed.path == "/api/perf_report":
            prof_client_path = str(os.environ.get("VHS_PROFILE_CLIENT", "")).strip()
            if not prof_client_path:
                self._send_json(
                    {"ok": False, "error": "Client profiling not enabled on server"}
                )
                return
            entry = {**payload, "_server_ts": __import__("time").time()}
            try:
                with open(prof_client_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
            except Exception as exc:
                self._send_error_json(f"Failed to write profile: {exc}")
                return
            self._send_json({"ok": True})
            return

        self._send_error_json("Not found", code=HTTPStatus.NOT_FOUND)

    def _handle_load_chapter(
        self, session: SessionState, payload: dict[str, Any]
    ) -> None:
        def fail(message: str) -> None:
            _set_load_progress(
                session,
                running=False,
                progress=0.0,
                message=str(message),
            )
            session.load_cancel_requested = False
            self._send_error_json(message)

        def cancelled() -> bool:
            if not bool(session.load_cancel_requested):
                return False
            fail("Load cancelled.")
            return True

        session.load_cancel_requested = False
        _set_load_progress(
            session,
            running=True,
            progress=1.0,
            message="Preparing chapter load...",
            sample_done=0,
            sample_total=0,
        )

        archive = str(payload.get("archive", session.archive) or "").strip()
        chapter = str(payload.get("chapter", session.chapter) or "").strip()
        if not archive or not chapter:
            fail("Archive and chapter are required.")
            return

        if cancelled():
            return
        _archive_state(session, archive, selected_title=chapter)
        chapter_obj = _find_chapter(session.chapters, chapter)
        if not chapter_obj:
            fail("Selected chapter was not found.")
            return

        default_start = int(chapter_obj.get("start_frame", 0))
        default_end = int(chapter_obj.get("end_frame", default_start + 1))

        try:
            iqr_k = _normalize_iqr_k(
                payload.get("iqr_k", session.iqr_k), default=session.iqr_k
            )
        except Exception:
            fail("Invalid numeric load settings.")
            return

        session.chapter = chapter
        session.start_frame, session.end_frame = _normalize_frame_span(
            default_start, default_end
        )
        session.debug_extract = bool(
            payload.get("debug_extract", session.debug_extract)
        )
        session.iqr_k = iqr_k
        session.force_all_frames_good = _normalize_payload_bool(
            payload.get("force_all_frames_good"),
            default=session.force_all_frames_good,
        )
        session.preview_video_path = ""
        session.chapter_audio_path = ""
        session.chapter_audio_key = ""
        _close_session_video_cap(session)
        session.fids = []
        session.b64 = []
        session.sigs = {}
        session.overrides = _chapter_bad_overrides(
            archive=session.archive,
            chapter_title=session.chapter,
            ch_start=session.start_frame,
            ch_end=session.end_frame,
        )
        session.gamma_default = 1.0
        session.gamma_ranges = []
        session.audio_sync_offset = 0.0
        session.audio_sync_audio_path = ""
        session.audio_sync_audio_key = ""
        session.people_entries = []
        session.subtitle_entries = []
        session.split_entries = []
        session.partial_fids = []
        session.partial_b64 = []
        session.partial_sigs = {"chroma": [], "noise": [], "tear": [], "wave": []}
        session.frame_source_video_path = ""
        session.frame_source_read_offset = 0
        session.load_meta_ready = False

        # Load metadata from disk early — fast TSV reads that don't require frame extraction.
        gamma_profile = get_gamma_profile_for_chapter(
            archive=session.archive,
            ch_start=session.start_frame,
            ch_end=session.end_frame,
        )
        session.gamma_default = _normalize_gamma_value(
            gamma_profile.get("default_gamma", 1.0), default=1.0
        )
        session.gamma_ranges = _normalize_gamma_ranges_payload(
            gamma_profile.get("ranges", []),
            ch_start=session.start_frame,
            ch_end=session.end_frame,
        )
        session.audio_sync_offset = get_audio_sync_offset_for_chapter(
            archive=session.archive,
            ch_start=session.start_frame,
            ch_end=session.end_frame,
        )
        session.people_entries = _load_people_entries_for_chapter(
            archive=session.archive,
            ch_start=session.start_frame,
            ch_end=session.end_frame,
        )
        session.subtitle_entries = _load_subtitle_entries_for_chapter(
            archive=session.archive,
            ch_start=session.start_frame,
            ch_end=session.end_frame,
        )
        session.auto_transcript = get_transcript_mode_for_chapter(
            archive=session.archive,
            chapter_title=session.chapter,
        )
        session.split_entries = _load_split_entries_for_chapter(
            archive=session.archive,
            chapter_title=session.chapter,
            ch_start=session.start_frame,
            ch_end=session.end_frame,
        )
        session.load_meta_ready = True

        video = _resolve_archive_video(session.archive)
        if not video:
            fail(f"No archive video found for '{session.archive}'.")
            return
        session.frame_source_video_path = str(video)
        session.frame_source_read_offset = 0

        if cancelled():
            return
        n_frames = max(1, int(session.end_frame) - int(session.start_frame))
        _set_load_progress(
            session,
            progress=4.0,
            message=f"Target frames: {int(n_frames)}",
            sample_done=0,
            sample_total=int(n_frames),
        )
        read_video = video
        frame_read_offset = 0

        debug_overlay = (
            bool(session.debug_extract)
            or _env_truthy(TUNER_DEBUG_EXTRACT_ENV)
            or _env_truthy(RENDER_DEBUG_EXTRACT_FRAME_NUMBERS_ENV)
        )

        # Build a frame-accurate review extract first; later frame IDs and
        # manual bad-frame edits are defined relative to this exact span.
        _set_load_progress(
            session, progress=8.0, message="Extracting chapter segment..."
        )
        if cancelled():
            return
        try:
            read_video_p, ex_err = _ensure_render_chapter_extract(
                source_video=video,
                archive=session.archive,
                chapter_title=session.chapter,
                ch_start=session.start_frame,
                ch_end=session.end_frame,
                debug_overlay=debug_overlay,
            )
        except Exception as exc:
            fail(f"Render extract failed: {type(exc).__name__}: {exc}")
            return
        if ex_err or read_video_p is None:
            fail(ex_err or "Render extract failed")
            return
        read_video = read_video_p
        frame_read_offset = session.start_frame
        session.frame_source_video_path = str(read_video)
        session.frame_source_read_offset = int(frame_read_offset)
        _set_load_progress(
            session, progress=28.0, message="Chapter extract ready; loading frames..."
        )

        if cancelled():
            return
        frame_target = max(1, int(n_frames))
        stage_start = 30.0
        stage_end = 92.0

        def _sample_progress(frac: float, desc: str | None = None) -> None:
            _ = desc
            try:
                f = float(frac)
            except Exception:
                f = 0.0
            f = max(0.0, min(1.0, f))
            done = max(0, min(frame_target, int(round(f * frame_target))))
            p = stage_start + f * (stage_end - stage_start)
            _set_load_progress(
                session,
                progress=p,
                message=f"Loading frames {done}/{frame_target}",
                sample_done=done,
                sample_total=frame_target,
            )

        def _sample_frame(
            fid: int,
            frame_b64: str,
            chroma: float,
            noise: float,
            tear: float,
            wave: float,
            _done: int,
            _total: int,
        ) -> None:
            session.partial_fids.append(int(fid))
            session.partial_b64.append(str(frame_b64 or ""))
            session.partial_sigs["chroma"].append(float(chroma))
            session.partial_sigs["noise"].append(float(noise))
            session.partial_sigs["tear"].append(float(tear))
            session.partial_sigs["wave"].append(float(wave))

        fids, b64, sigs, err = extract_frames(
            str(read_video),
            session.start_frame,
            session.end_frame,
            int(n_frames),
            session.archive,
            session.chapter,
            include_thumbs=False,
            frame_read_offset=frame_read_offset,
            progress=_sample_progress,
            should_cancel=lambda: bool(session.load_cancel_requested),
            frame_callback=_sample_frame,
        )
        if err or fids is None or b64 is None or sigs is None:
            fail(err or "Failed to extract frames.")
            return
        _set_load_progress(
            session,
            progress=95.0,
            message=f"Processing frames {frame_target}/{frame_target}",
            sample_done=frame_target,
            sample_total=frame_target,
        )

        session.fids = [int(x) for x in fids]
        session.b64 = list(b64)
        session.sigs = dict(sigs)

        details = {
            "archive": session.archive,
            "chapter": session.chapter,
            "start_frame": session.start_frame,
            "end_frame": session.end_frame,
            "force_all_frames_good": bool(session.force_all_frames_good),
            "chapter_frame_count": int(session.end_frame - session.start_frame),
            "loaded_count": int(len(session.fids)),
            "contact_sheet": _contact_sheet_config_payload(session),
            "extract_cache": str(
                _chapter_extract_cache_path(
                    archive=session.archive,
                    chapter_title=session.chapter,
                    ch_start=session.start_frame,
                    ch_end=session.end_frame,
                    debug_overlay=debug_overlay,
                    source_video=video,
                ).parent.name
            ),
            "gamma_profile": {
                "default_gamma": float(session.gamma_default),
                "ranges": list(session.gamma_ranges),
                "source": str(gamma_profile.get("source", "default")),
            },
            "audio_sync_profile": {
                "offset_seconds": float(session.audio_sync_offset),
            },
            "people_profile": {
                "entries": list(session.people_entries),
                "source": "people_tsv",
            },
            "subtitles_profile": {
                "entries": list(session.subtitle_entries),
                "source": "subtitles_tsv",
            },
            "auto_transcript": str(session.auto_transcript),
            "split_profile": {
                "entries": list(session.split_entries),
                "source": "chapters_tsv",
            },
        }

        review = _build_review_payload(
            session,
            include_images=False,
        )
        _set_load_progress(
            session,
            running=False,
            progress=100.0,
            message=f"Loaded {len(session.fids)} frame(s).",
            sample_done=len(session.fids),
            sample_total=max(1, int(n_frames)),
        )
        session.load_cancel_requested = False
        self._send_json({"ok": True, "review": review, "settings": details})

    def _handle_toggle_frame(self, session: SessionState, fid: int) -> None:
        if bool(session.force_all_frames_good):
            self._send_error_json(
                "Disable 'Force all frames good' before editing frame statuses."
            )
            return
        fid_i = int(fid)
        final_ids = {int(x) for x in session.fids}
        partial_ids = {int(x) for x in session.partial_fids}
        if fid_i not in final_ids and fid_i not in partial_ids:
            self._send_error_json("Frame is not in the loaded set.")
            return

        if session.fids and session.sigs and fid_i in final_ids:
            scores = combined_score(
                session.sigs, session.wc, session.wn, session.wt, session.ww
            )
            thr = float(
                compute_threshold(
                    scores, session.t_mode, session.iqr_k, session.tval, session.bpct
                )
            )
            index = {int(x): i for i, x in enumerate(session.fids)}
            pos = index[fid_i]
            score = float(scores[pos])
            effective, _src = _frame_status(session, fid_i, score, thr)
        else:
            partial_review = _build_partial_review_payload(
                session, include_images=False
            )
            current = next(
                (f for f in partial_review["frames"] if int(f["fid"]) == fid_i), None
            )
            if not current:
                self._send_error_json("Frame is not available yet.")
                return
            effective = "bad" if str(current.get("status")) == "bad" else "good"

        session.overrides[fid_i] = "good" if effective == "bad" else "bad"

        if session.fids and session.sigs:
            frame_state = _build_review_payload(session, include_images=False)
        else:
            frame_state = _build_partial_review_payload(session, include_images=False)
        updated = next(
            (f for f in frame_state["frames"] if int(f["fid"]) == fid_i), None
        )
        self._send_json({"ok": True, "frame": updated, "review": frame_state})

    def _handle_set_frame_range(
        self, session: SessionState, start_fid: int, end_fid: int, status: str
    ) -> None:
        if bool(session.force_all_frames_good):
            self._send_error_json(
                "Disable 'Force all frames good' before editing frame statuses."
            )
            return
        lo = int(min(int(start_fid), int(end_fid)))
        hi = int(max(int(start_fid), int(end_fid)))
        target_status = "good" if str(status).strip().lower() == "good" else "bad"
        if session.fids and session.sigs:
            current = _build_review_payload(session, include_images=False)
        else:
            current = _build_partial_review_payload(session, include_images=False)

        changed = 0
        for frame in list(current.get("frames", [])):
            try:
                fid_i = int(frame.get("fid"))
            except Exception:
                continue
            if fid_i < lo or fid_i > hi:
                continue
            session.overrides[fid_i] = target_status
            changed += 1

        if changed <= 0:
            self._send_error_json(
                "No loaded frames are currently available in that range."
            )
            return

        if session.fids and session.sigs:
            review = _build_review_payload(session, include_images=False)
        else:
            review = _build_partial_review_payload(session, include_images=False)
        self._send_json(
            {
                "ok": True,
                "review": review,
                "range": {"start_fid": lo, "end_fid": hi},
                "status": target_status,
                "updated_count": int(changed),
            }
        )

    def _handle_set_force_all_good(
        self, session: SessionState, payload: dict[str, Any] | None = None
    ) -> None:
        payload = payload or {}
        enabled = _normalize_payload_bool(
            payload.get("enabled", payload.get("force_all_frames_good", False)),
            default=session.force_all_frames_good,
        )
        session.force_all_frames_good = bool(enabled)

        if session.fids and session.sigs:
            # Fast path: compute stats without building the full per-frame list.
            # The client updates frame statuses locally using its pre-force snapshot.
            scores = combined_score(
                session.sigs, session.wc, session.wn, session.wt, session.ww
            )
            thr = float(
                compute_threshold(
                    scores, session.t_mode, session.iqr_k, session.tval, session.bpct
                )
            )
            session.threshold = thr
            total = len(session.fids)
            if enabled:
                bad = 0
            else:
                scores_arr = np.asarray(scores, dtype=np.float64)
                bad = int(np.sum(scores_arr >= thr))
                fids_list = session.fids
                for fid_key, ov in session.overrides.items():
                    fid = int(fid_key)
                    idx = bisect.bisect_left(fids_list, fid)
                    if idx < total and fids_list[idx] == fid:
                        auto_is_bad = bool(float(scores_arr[idx]) >= thr)
                        if ov == "good" and auto_is_bad:
                            bad -= 1
                        elif ov == "bad" and not auto_is_bad:
                            bad += 1
                bad = max(0, bad)
            review = {
                "threshold": round(thr, 4),
                "stats": {
                    "total": total,
                    "bad": int(bad),
                    "good": int(total - bad),
                    "shown": total,
                    "overrides": int(len(session.overrides)),
                },
                "force_all_frames_good": bool(session.force_all_frames_good),
                "frames": None,
            }
        elif session.partial_fids:
            review = _build_partial_review_payload(session, include_images=False)
        else:
            review = {
                "threshold": 0.0,
                "stats": {"total": 0, "bad": 0, "good": 0, "shown": 0, "overrides": 0},
                "force_all_frames_good": bool(session.force_all_frames_good),
                "frames": None,
            }
        self._send_json({"ok": True, "review": review})

    def _run_cmd(self, cmd: list[Any], label: str) -> tuple[bool, str]:
        proc = subprocess.run(
            [str(x) for x in cmd],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return True, ""
        detail = (proc.stderr or proc.stdout or "").strip()
        if not detail:
            detail = f"{label} failed with exit code {int(proc.returncode)}."
        else:
            detail = f"{label} failed: {detail}"
        return False, detail

    def _run_cmd_with_progress(
        self,
        cmd: list[Any],
        label: str,
        *,
        on_frame: Any | None = None,
    ) -> tuple[bool, str]:
        parts = [str(x) for x in cmd]
        if parts:
            ffmpeg_name = Path(parts[0]).name.lower()
            if ffmpeg_name in {"ffmpeg", "ffmpeg.exe"} and "-progress" not in parts:
                parts = [
                    parts[0],
                    "-progress",
                    "pipe:2",
                    "-stats_period",
                    "0.5",
                    *parts[1:],
                ]

        proc = subprocess.Popen(
            parts,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        err_lines: list[str] = []
        try:
            if proc.stderr is not None:
                for raw in proc.stderr:
                    line = str(raw or "").strip()
                    if line:
                        err_lines.append(line)
                        if len(err_lines) > 80:
                            err_lines = err_lines[-80:]
                    m = re.match(r"^frame\s*=\s*(\d+)$", line)
                    if m and on_frame is not None:
                        try:
                            on_frame(int(m.group(1)))
                        except Exception:
                            pass
        finally:
            rc = proc.wait()

        if rc == 0:
            return True, ""
        detail = "\n".join(err_lines).strip()
        if not detail:
            detail = f"{label} failed with exit code {int(rc)}."
        else:
            detail = f"{label} failed: {detail}"
        return False, detail

    def _handle_preview_render(
        self, session: SessionState, payload: dict[str, Any] | None = None
    ) -> None:
        def fail(message: str) -> None:
            _set_preview_progress(
                session,
                running=False,
                progress=0.0,
                message=str(message),
            )
            self._send_error_json(message)

        if not session.fids or not session.sigs:
            fail("No loaded chapter data yet.")
            return
        if not session.archive or not session.chapter:
            fail("Archive and chapter context are missing.")
            return

        payload = payload or {}
        preview_mode = str(payload.get("preview_mode", "") or "").strip().lower()
        if "force_all_frames_good" in payload:
            session.force_all_frames_good = _normalize_payload_bool(
                payload.get("force_all_frames_good"),
                default=session.force_all_frames_good,
            )
        apply_freeze = True
        apply_gamma = True
        if preview_mode == "review":
            apply_gamma = False
        elif preview_mode == "gamma":
            apply_freeze = False
        elif preview_mode == "summary":
            apply_freeze = True
            apply_gamma = True

        raw_gamma_profile = payload.get("gamma_profile")
        if raw_gamma_profile is None:
            raw_gamma_profile = payload.get("gamma")
        if isinstance(raw_gamma_profile, dict):
            session.gamma_default = _normalize_gamma_value(
                raw_gamma_profile.get("default_gamma", session.gamma_default),
                default=session.gamma_default,
            )
            session.gamma_ranges = _normalize_gamma_ranges_payload(
                raw_gamma_profile.get("ranges", session.gamma_ranges),
                ch_start=session.start_frame,
                ch_end=session.end_frame,
            )

        try:
            from vhs_pipeline.render_pipeline import (
                BADFRAME_SOURCE_CLEARANCE,
                assert_expected_frame_count,
                local_bad_frames_to_repairs,
                make_create_avs,
                make_deinterlace,
                make_deinterlace_ffmpeg_fallback,
                make_freeze_only_avs,
                make_gamma_only_avs,
                make_render_avs_ffv1,
            )
        except Exception as exc:
            fail(f"Preview render is unavailable: {type(exc).__name__}: {exc}")
            return

        proxy_video = ARCHIVE_DIR / f"{session.archive}_proxy.mp4"
        archive_video = ARCHIVE_DIR / f"{session.archive}.mkv"
        if proxy_video.exists():
            source_video = proxy_video
            source_label = "proxy"
        elif archive_video.exists():
            source_video = archive_video
            source_label = "archive (proxy missing)"
        else:
            fail(f"No source video found for '{session.archive}'.")
            return

        start_frame, end_frame = _normalize_frame_span(
            session.start_frame, session.end_frame
        )
        chapter_len = max(1, int(end_frame) - int(start_frame))
        debug_overlay = (
            bool(session.debug_extract)
            or _env_truthy(TUNER_DEBUG_EXTRACT_ENV)
            or _env_truthy(RENDER_DEBUG_EXTRACT_FRAME_NUMBERS_ENV)
        )
        # Preview generation reuses the same frame-accurate chapter extract as
        # the review flow so frame indices stay aligned across the tuner.
        extracted, ex_err = _ensure_render_chapter_extract(
            source_video=source_video,
            archive=session.archive,
            chapter_title=session.chapter,
            ch_start=start_frame,
            ch_end=end_frame,
            debug_overlay=debug_overlay,
        )
        if ex_err or extracted is None:
            fail(ex_err or "Failed to extract preview chapter segment.")
            return

        bad_global = [
            int(fid)
            for fid in _selected_bad_frame_ids(session)
            if int(start_frame) <= int(fid) < int(end_frame)
        ]
        local_bad = (
            [int(fid) - int(start_frame) for fid in bad_global] if apply_freeze else []
        )
        local_repairs = local_bad_frames_to_repairs(local_bad) if local_bad else []

        preview_root = PROJECT_ROOT / "tmp" / "plain_html_wizard_preview"
        preview_dir = preview_root / (
            f"{session.archive}__{slugify(session.chapter)}__{int(start_frame)}_{int(end_frame)}"
        )
        preview_dir.mkdir(parents=True, exist_ok=True)

        freeze_avs = preview_dir / "freeze.avs"
        filter_avs = preview_dir / "script.avs"
        repaired_extracted = preview_dir / "repaired_extracted.mkv"
        qtgmc = preview_dir / "qtgmc.mkv"
        preview_video = preview_dir / "preview_render.mp4"

        filter_script = METADATA_DIR / session.archive / "filter.avs"
        chapter_filter_script = (
            METADATA_DIR / session.archive / f"{session.chapter}.avs"
        )
        if chapter_filter_script.exists():
            filter_script = chapter_filter_script

        freeze_input = extracted
        used_non_windows_fallback = False
        gamma_only_mode = preview_mode == "gamma"
        windows_filter = bool(
            sys.platform == "win32"
            and apply_gamma
            and (gamma_only_mode or filter_script.exists())
        )
        windows_freeze = bool(sys.platform == "win32" and bool(local_bad))
        stage_names: list[str] = []
        if windows_freeze:
            stage_names.append("Applying FreezeFrame repairs")
        if windows_filter:
            stage_names.append(
                "Applying gamma correction"
                if gamma_only_mode
                else "Deinterlacing/filtering"
            )
        elif sys.platform != "win32":
            stage_names.append("Fallback deinterlacing")
        stage_names.append("Encoding preview")
        total_stages = max(1, len(stage_names))
        total_frames_all = max(1, chapter_len * total_stages)

        _set_preview_progress(
            session,
            running=True,
            progress=1.0,
            message="Preparing preview render...",
            frame_done=0,
            frame_total=total_frames_all,
        )

        def _set_stage_progress(
            stage_idx: int, frame_done: int, stage_label: str
        ) -> None:
            done = max(0, min(chapter_len, int(frame_done)))
            overall_done = min(total_frames_all, (stage_idx * chapter_len) + done)
            frac = float(done) / float(max(1, chapter_len))
            pct = ((float(stage_idx) + frac) / float(total_stages)) * 100.0
            _set_preview_progress(
                session,
                running=True,
                progress=max(1.0, min(99.5, pct)),
                message=f"{stage_label}... ({done}/{chapter_len} frames)",
                frame_done=overall_done,
                frame_total=total_frames_all,
            )

        stage_idx = 0

        try:
            if sys.platform == "win32":
                if local_bad:
                    stage_label = stage_names[stage_idx]
                    freeze_script = make_freeze_only_avs(
                        extracted,
                        bad_source_frames=local_bad,
                        bad_repair_ranges=local_repairs,
                        chapter_start_frame=start_frame,
                        chapter_end_frame=end_frame,
                        source_clearance=BADFRAME_SOURCE_CLEARANCE,
                    )
                    freeze_avs.write_text(freeze_script, encoding="ascii")
                    ok, detail = self._run_cmd_with_progress(
                        make_render_avs_ffv1(freeze_avs, extracted, repaired_extracted),
                        "Preview freeze stage",
                        on_frame=lambda n: _set_stage_progress(
                            stage_idx, n, stage_label
                        ),
                    )
                    if not ok:
                        fail(detail)
                        return
                    _set_stage_progress(stage_idx, chapter_len, stage_label)
                    stage_idx += 1
                    assert_expected_frame_count(
                        repaired_extracted,
                        chapter_len,
                        f"preview repaired chapter '{session.chapter}'",
                    )
                    freeze_input = repaired_extracted

                if windows_filter:
                    stage_label = stage_names[stage_idx]
                    gamma_default = _normalize_gamma_value(
                        session.gamma_default, default=1.0
                    )
                    gamma_ranges = _normalize_gamma_ranges_payload(
                        session.gamma_ranges,
                        ch_start=start_frame,
                        ch_end=end_frame,
                    )
                    if gamma_only_mode:
                        script_text = make_gamma_only_avs(
                            freeze_input,
                            chapter_start_frame=start_frame,
                            chapter_end_frame=end_frame,
                            gamma_default=gamma_default,
                            gamma_ranges=gamma_ranges,
                        )
                    else:
                        script_text = make_create_avs(
                            freeze_input,
                            filter_script,
                            bad_source_frames=[],
                            bad_repair_ranges=[],
                            chapter_start_frame=start_frame,
                            chapter_end_frame=end_frame,
                            gamma_default=gamma_default,
                            gamma_ranges=gamma_ranges,
                            no_bob=False,
                            source_clearance=0,
                        )
                    filter_avs.write_text(script_text, encoding="ascii")
                    stage_cmd_label = (
                        "Preview gamma stage"
                        if gamma_only_mode
                        else "Preview deinterlace stage"
                    )
                    ok, detail = self._run_cmd_with_progress(
                        make_deinterlace(filter_avs, freeze_input, qtgmc),
                        stage_cmd_label,
                        on_frame=lambda n: _set_stage_progress(
                            stage_idx, n, stage_label
                        ),
                    )
                    if not ok:
                        fail(detail)
                        return
                    _set_stage_progress(stage_idx, chapter_len, stage_label)
                    stage_idx += 1
                else:
                    shutil.copy2(freeze_input, qtgmc)
            else:
                used_non_windows_fallback = True
                stage_label = stage_names[stage_idx]
                ok, detail = self._run_cmd_with_progress(
                    make_deinterlace_ffmpeg_fallback(extracted, qtgmc, no_bob=False),
                    "Preview fallback deinterlace stage",
                    on_frame=lambda n: _set_stage_progress(stage_idx, n, stage_label),
                )
                if not ok:
                    fail(detail)
                    return
                _set_stage_progress(stage_idx, chapter_len, stage_label)
                stage_idx += 1

            assert_expected_frame_count(
                qtgmc,
                chapter_len,
                f"preview qtgmc chapter '{session.chapter}'",
            )
        except Exception as exc:
            fail(f"Preview render failed: {type(exc).__name__}: {exc}")
            return

        stage_label = stage_names[
            stage_idx if stage_idx < len(stage_names) else (len(stage_names) - 1)
        ]
        ok, detail = self._run_cmd_with_progress(
            [
                FFMPEG_BIN,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(qtgmc),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode:v:0",
                "passthrough",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-ar",
                "48000",
                "-ac",
                "1",
                "-movflags",
                "+faststart",
                "-y",
                str(preview_video),
            ],
            "Preview encode stage",
            on_frame=lambda n: _set_stage_progress(stage_idx, n, stage_label),
        )
        if not ok:
            fail(detail)
            return
        _set_stage_progress(stage_idx, chapter_len, stage_label)

        session.preview_video_path = str(preview_video.resolve())
        mode_desc = (
            preview_mode
            if preview_mode
            in {"review", "gamma", "people", "subtitles", "split", "summary"}
            else "combined"
        )
        msg = (
            f"Preview render ready for {session.chapter}: "
            f"mode={mode_desc}, freeze={'on' if apply_freeze else 'off'}, gamma={'on' if apply_gamma else 'off'}. "
            f"{len(local_bad)} loaded bad frame(s) applied from current review state. "
            f"Source: {source_label}."
        )
        if used_non_windows_fallback and local_bad:
            msg += " Note: non-Windows fallback cannot apply AviSynth FreezeFrame repair logic."
        _set_preview_progress(
            session,
            running=False,
            progress=100.0,
            message="Preview render complete.",
            frame_done=total_frames_all,
            frame_total=total_frames_all,
        )
        self._send_json(
            {
                "ok": True,
                "message": msg,
                "preview_path": str(preview_video),
                "preview_url": "/api/preview_video",
                "preview_page_url": "/preview",
                "bad_frame_count": int(len(local_bad)),
            }
        )

    def _handle_people_prefill_cast(
        self, session: SessionState, payload: dict[str, Any] | None = None
    ) -> None:
        if not session.archive or not session.chapter:
            self._send_error_json("Load a chapter before running Cast prefill.")
            return
        payload = payload or {}

        chapter_duration = max(
            0.0,
            _frame_to_seconds(session.end_frame)
            - _frame_to_seconds(session.start_frame),
        )
        mode = str(payload.get("mode") or "replace").strip().lower()
        if mode not in {"replace", "append"}:
            mode = "replace"
        cast_store_raw = str(payload.get("cast_store_dir") or "").strip()
        cast_store_dir = (
            Path(cast_store_raw) if cast_store_raw else DEFAULT_CAST_STORE_DIR
        )
        min_quality = payload.get("min_quality", 0.40)
        min_name_hits = payload.get("min_name_hits", 1)
        try:
            result = prefill_people_from_cast(
                archive=session.archive,
                chapter_title=session.chapter,
                cast_store_dir=cast_store_dir,
                min_quality=float(min_quality),
                min_name_hits=max(1, int(min_name_hits)),
            )
        except Exception as exc:
            self._send_error_json(f"Cast prefill failed: {type(exc).__name__}: {exc}")
            return

        generated = _normalize_people_entries_payload(
            list(result.entries or []),
            chapter_duration_seconds=chapter_duration,
        )
        if generated:
            if mode == "append":
                merged = _normalize_people_entries_payload(
                    [*(session.people_entries or []), *generated],
                    chapter_duration_seconds=chapter_duration,
                )
            else:
                merged = generated
            session.people_entries = list(merged)
            message = (
                f"Cast prefill added {len(generated)} entr"
                f"{'y' if len(generated) == 1 else 'ies'} ({mode})."
            )
        else:
            message = "Cast prefill found no confident matches for this chapter."

        self._send_json(
            {
                "ok": True,
                "message": message,
                "generated_count": int(len(generated)),
                "mode": mode,
                "cast_store_dir": str(Path(cast_store_dir)),
                "stats": dict(result.stats or {}),
                "people_profile": {
                    "entries": list(session.people_entries),
                    "source": "cast_prefill",
                },
            }
        )

    def _handle_subtitles_generate(
        self, session: SessionState, payload: dict[str, Any] | None = None
    ) -> None:
        def fail(message: str) -> None:
            _set_subtitles_progress(
                session,
                running=False,
                progress=0.0,
                message=str(message),
                segment_done=0,
                segment_total=0,
            )
            session.subtitles_cancel_requested = False
            self._send_error_json(message)

        def cancelled() -> bool:
            if not bool(session.subtitles_cancel_requested):
                return False
            fail("Subtitle generation cancelled.")
            return True

        if not session.archive or not session.chapter:
            fail("Load a chapter before generating subtitles.")
            return
        if int(session.end_frame) <= int(session.start_frame):
            fail("Invalid chapter frame span for subtitle generation.")
            return
        if bool(session.subtitles_running):
            self._send_error_json(
                "Subtitle generation is already running for this session."
            )
            return
        payload = payload or {}
        mode = str(payload.get("mode") or "replace").strip().lower()
        if mode not in {"replace", "append"}:
            mode = "replace"

        source_video = _resolve_archive_video(session.archive)
        if not source_video:
            fail(f"No archive video found for '{session.archive}'.")
            return

        session.subtitles_cancel_requested = False
        _set_subtitles_progress(
            session,
            running=True,
            progress=1.0,
            message="Preparing Whisper subtitle generation...",
            segment_done=0,
            segment_total=0,
        )
        if cancelled():
            return

        try:
            model = _load_whisper_model()
            transcribe_module = _load_whisper_transcribe_module()
        except Exception as exc:
            fail(f"Subtitle generation unavailable: {type(exc).__name__}: {exc}")
            return
        if cancelled():
            return

        chapter_duration = max(
            0.0,
            _frame_to_seconds(session.end_frame)
            - _frame_to_seconds(session.start_frame),
        )
        prompt_text = _subtitle_prompt_from_people_entries(
            _normalize_people_entries_payload(
                session.people_entries,
                chapter_duration_seconds=chapter_duration,
            )
        )
        start_sec = _frame_to_seconds(session.start_frame)
        end_sec = _frame_to_seconds(session.end_frame)

        _set_subtitles_progress(
            session,
            progress=5.0,
            message="Extracting chapter audio for Whisper...",
            segment_done=0,
            segment_total=0,
        )

        try:
            with tempfile.TemporaryDirectory(prefix="vhs_subtitles_") as tmp_dir:
                audio_path = Path(tmp_dir) / "chapter_audio.wav"
                cmd = make_extract_audio(
                    source_video,
                    audio_path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                )
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0 or not audio_path.exists():
                    detail = (proc.stderr or proc.stdout or "").strip()
                    raise RuntimeError(detail or "ffmpeg audio extraction failed.")
                if bool(session.subtitles_cancel_requested):
                    raise _SubtitlesCancelledError("Subtitle generation cancelled.")

                segment_total = 0
                try:
                    with wave.open(str(audio_path), "rb") as wf:
                        sample_rate = int(wf.getframerate() or 0)
                        sample_count = int(wf.getnframes() or 0)
                    if sample_rate > 0 and sample_count > 0:
                        duration_sec = float(sample_count) / float(sample_rate)
                        segment_total = max(1, int(round(duration_sec * 50.0)))
                except Exception:
                    segment_total = 0

                _set_subtitles_progress(
                    session,
                    progress=20.0,
                    message="Transcribing audio with Whisper...",
                    segment_done=0,
                    segment_total=segment_total,
                )

                progress_prefix = "Transcribing audio with Whisper..."
                tqdm_module = getattr(transcribe_module, "tqdm", None)
                original_tqdm = (
                    getattr(tqdm_module, "tqdm", None)
                    if tqdm_module is not None
                    else None
                )

                class _SubtitlesProgressBar:
                    def __init__(self, *args: Any, **kwargs: Any) -> None:
                        self._inner = original_tqdm(*args, **kwargs)
                        total_raw = getattr(self._inner, "total", None)
                        try:
                            total_val = float(total_raw)
                        except Exception:
                            total_val = float(segment_total or 0)
                        self._total = (
                            max(1.0, total_val)
                            if total_val > 0
                            else max(1.0, float(segment_total or 1))
                        )
                        _set_subtitles_progress(
                            session,
                            progress=20.0,
                            message=progress_prefix,
                            segment_done=0,
                            segment_total=max(1, int(round(self._total))),
                        )

                    def __enter__(self) -> "_SubtitlesProgressBar":
                        if hasattr(self._inner, "__enter__"):
                            self._inner.__enter__()
                        return self

                    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
                        if hasattr(self._inner, "__exit__"):
                            return self._inner.__exit__(exc_type, exc, tb)
                        return False

                    def update(self, n: int = 1) -> Any:
                        out = self._inner.update(n)
                        done_raw = getattr(self._inner, "n", 0)
                        try:
                            done = max(0.0, float(done_raw))
                        except Exception:
                            done = 0.0
                        frac = min(1.0, done / max(1.0, self._total))
                        _set_subtitles_progress(
                            session,
                            progress=20.0 + (frac * 75.0),
                            message=progress_prefix,
                            segment_done=max(0, int(round(done))),
                            segment_total=max(1, int(round(self._total))),
                        )
                        if bool(session.subtitles_cancel_requested):
                            raise _SubtitlesCancelledError(
                                "Subtitle generation cancelled."
                            )
                        return out

                    def close(self) -> Any:
                        if hasattr(self._inner, "close"):
                            return self._inner.close()
                        return None

                    def __iter__(self) -> Any:
                        return iter(self._inner)

                    def __getattr__(self, name: str) -> Any:
                        return getattr(self._inner, name)

                with _WHISPER_TQDM_PATCH_LOCK:
                    if callable(original_tqdm):
                        setattr(tqdm_module, "tqdm", _SubtitlesProgressBar)
                    try:
                        result = whisper_transcribe(
                            model, audio_path, prompt_text=prompt_text
                        )
                    finally:
                        if callable(original_tqdm):
                            setattr(tqdm_module, "tqdm", original_tqdm)
                if bool(session.subtitles_cancel_requested):
                    raise _SubtitlesCancelledError("Subtitle generation cancelled.")
        except _SubtitlesCancelledError:
            fail("Subtitle generation cancelled.")
            return
        except Exception as exc:
            fail(f"Subtitle generation failed: {type(exc).__name__}: {exc}")
            return

        _set_subtitles_progress(
            session,
            progress=97.0,
            message="Formatting generated subtitle segments...",
            segment_done=max(0, int(session.subtitles_segment_total)),
            segment_total=max(0, int(session.subtitles_segment_total)),
        )

        generated = _normalize_subtitle_entries_payload(
            subtitle_entries_from_whisper_result(result),
            chapter_duration_seconds=chapter_duration,
        )
        if generated:
            if mode == "append":
                merged = _normalize_subtitle_entries_payload(
                    [*(session.subtitle_entries or []), *generated],
                    chapter_duration_seconds=chapter_duration,
                )
            else:
                merged = generated
            session.subtitle_entries = list(merged)
            message = (
                f"Generated {len(generated)} subtitle entr"
                f"{'y' if len(generated) == 1 else 'ies'} ({mode})."
            )
        else:
            message = "Whisper returned no subtitle segments for this chapter."

        _set_subtitles_progress(
            session,
            running=False,
            progress=100.0,
            message="Whisper subtitle generation complete.",
            segment_done=max(0, int(session.subtitles_segment_total)),
            segment_total=max(0, int(session.subtitles_segment_total)),
        )
        session.subtitles_cancel_requested = False

        self._send_json(
            {
                "ok": True,
                "message": message,
                "generated_count": int(len(generated)),
                "mode": mode,
                "subtitles_profile": {
                    "entries": list(session.subtitle_entries),
                    "source": "whisper_generate",
                },
                "subtitles": {
                    "entries": list(session.subtitle_entries),
                    "source": "whisper_generate",
                },
            }
        )

    def _handle_set_auto_transcript(
        self, session: SessionState, payload: dict[str, Any] | None = None
    ) -> None:
        if not session.archive or not session.chapter:
            self._send_error_json("No loaded chapter data yet.")
            return
        raw = (payload or {}).get("auto_transcript", "off")
        mode = "on" if str(raw).strip().lower() in {"on", "true", "1", "yes"} else "off"
        try:
            update_chapter_transcript_in_chapters_tsv(
                session.archive,
                session.chapter,
                transcript=mode,
            )
        except Exception as exc:
            self._send_error_json(str(exc))
            return
        session.auto_transcript = mode
        self._send_json({"ok": True, "auto_transcript": mode})

    def _handle_save(
        self, session: SessionState, payload: dict[str, Any] | None = None
    ) -> None:
        if not session.fids:
            self._send_error_json("No loaded chapter data yet.")
            return

        _apply_profiles_from_payload(session, payload)
        try:
            (
                out_path,
                gamma_path,
                split_path,
                count,
                analyzed,
                people_count,
                subtitle_count,
                split_count,
            ) = _persist_session_progress(session)
        except Exception as exc:
            self._send_error_json(str(exc))
            return
        gamma_count = len(session.gamma_ranges)
        people_path = METADATA_DIR / str(session.archive or "").strip() / "people.tsv"
        subtitles_path = (
            METADATA_DIR / str(session.archive or "").strip() / "subtitles.tsv"
        )
        chapters_path = (
            METADATA_DIR / str(session.archive or "").strip() / "chapters.tsv"
        )

        archive_state = _archive_state(
            session, session.archive, selected_title=session.chapter
        )
        self._send_json(
            {
                "ok": True,
                "message": (
                    f"Saved BAD_FRAMES for {session.chapter} "
                    f"({int(analyzed)} analyzed, {int(count)} bad). "
                    f"Saved gamma ranges: {int(gamma_count)}. "
                    f"Saved people entries: {int(people_count)}. "
                    f"Saved subtitle entries: {int(subtitle_count)}. "
                    f"Saved split entries: {int(split_count)}."
                ),
                "metadata_path": (
                    str(
                        split_path
                        or gamma_path
                        or out_path
                        or chapters_path
                        or subtitles_path
                        or people_path
                    )
                    if (
                        split_path
                        or gamma_path
                        or out_path
                        or chapters_path
                        or subtitles_path
                        or people_path
                    )
                    else ""
                ),
                "archive_state": archive_state,
            }
        )

    def _handle_save_progress(
        self, session: SessionState, payload: dict[str, Any] | None = None
    ) -> None:
        if not session.fids:
            self._send_error_json("No loaded chapter data yet.")
            return
        _apply_profiles_from_payload(session, payload)
        try:
            (
                out_path,
                gamma_path,
                split_path,
                count,
                analyzed,
                people_count,
                subtitle_count,
                split_count,
            ) = _persist_session_progress(session)
        except Exception as exc:
            self._send_error_json(str(exc))
            return
        self._send_json(
            {
                "ok": True,
                "message": (
                    f"Progress saved for {session.chapter}: "
                    f"BAD_FRAMES {int(count)}/{int(analyzed)}, "
                    f"gamma ranges {int(len(session.gamma_ranges))}, "
                    f"people entries {int(people_count)}, "
                    f"subtitle entries {int(subtitle_count)}, "
                    f"split entries {int(split_count)}."
                ),
                "metadata_path": str(
                    split_path
                    or (
                        METADATA_DIR
                        / str(session.archive or "").strip()
                        / "chapters.tsv"
                    )
                    or gamma_path
                    or out_path
                    or (
                        METADATA_DIR
                        / str(session.archive or "").strip()
                        / "subtitles.tsv"
                    )
                    or (
                        METADATA_DIR / str(session.archive or "").strip() / "people.tsv"
                    )
                ),
            }
        )


def run(host: str = "0.0.0.0", port: int = 8092) -> None:
    import socket

    server = ThreadingHTTPServer((host, int(port)), WizardHandler)
    server.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"VHS Tuner running at http://{host}:{port}")
    prof_path = str(os.environ.get("VHS_PROFILE_FRAMES", "")).strip()
    if prof_path:
        print(f"Frame profiling enabled — writing to: {prof_path}")
    prof_client_path = str(os.environ.get("VHS_PROFILE_CLIENT", "")).strip()
    if prof_client_path:
        print(f"Client profiling enabled — writing to: {prof_client_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
