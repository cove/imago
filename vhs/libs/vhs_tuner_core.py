# Core logic extracted from vhs_tuner.py

from __future__ import annotations

import csv
import base64
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# -- Project paths -------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name in {"scripts", "libs"} else _HERE
sys.path.insert(0, str(PROJECT_ROOT))

ARCHIVE_DIR = PROJECT_ROOT / "../../Archive"
METADATA_DIR = PROJECT_ROOT / "metadata"
FPS = 30000 / 1001
TUNER_CACHE_ROOT = Path(os.environ.get("VHS_TUNER_CACHE_DIR") or (Path(tempfile.gettempdir()) / "vhs_tuner_cache"))
TUNER_EXTRACT_DIR = TUNER_CACHE_ROOT / "extracts"
TUNER_FRAME_CACHE_DIR = TUNER_CACHE_ROOT / "frame_samples"
TUNER_DEBUG_EXTRACT_ENV = "VHS_TUNER_DEBUG_EXTRACT_FRAMES"
RENDER_DEBUG_EXTRACT_FRAME_NUMBERS_ENV = "RENDER_DEBUG_EXTRACT_FRAME_NUMBERS"
TUNER_FRAME_CACHE_VERSION = 1
_CACHE_SIGNAL_KEYS = ("chroma", "noise", "tear", "wave")
_LAST_CACHE_CLEANUP_TS = 0.0

from common import (
    FFPROBE_BIN,
    chapter_frame_bounds,
    combined_score,
    compute_threshold,
    get_bad_frames_for_chapter,
    make_frame_accurate_extract_chapter,
    parse_chapters,
    replace_chapter_bad_frames_in_render_settings,
)

# ===============================================================================
# Chapter / metadata helpers
# ===============================================================================


def load_archive_chapters(path: Path) -> list[dict]:
    # Keep chapter frame mapping identical to the render pipeline.
    path = Path(path)
    archive = str(path.parent.name)
    _ffm, chapters = parse_chapters(path)
    result = []
    for ch in chapters:
        title = str(ch.get("title", "Untitled"))
        start_sec = float(ch.get("start", 0.0))
        end_sec = float(ch.get("end", 0.0))
        start_frame, end_frame = chapter_frame_bounds(ch, fps_num=30000, fps_den=1001)
        result.append(
            {
                "title": title,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "bad_frames": get_bad_frames_for_chapter(archive, title),
            }
        )
    return result


# Backward-compatible alias for older callers.
parse_ffmetadata_chapters = load_archive_chapters


def _metadata_archive_dir(archive: str) -> Path:
    return METADATA_DIR / str(archive or "").strip()


def _chapters_tsv_path(archive: str) -> Path:
    return _metadata_archive_dir(archive) / "chapters.tsv"


def _chapters_ffmetadata_path(archive: str) -> Path:
    return _metadata_archive_dir(archive) / "chapters.ffmetadata"


def _parse_int_value(raw: object) -> int | None:
    text = str(raw if raw is not None else "").strip()
    if not text or not re.fullmatch(r"-?\d+", text):
        return None
    try:
        return int(text)
    except Exception:
        return None


def _parse_float_value(raw: object) -> float | None:
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _parse_timebase_value(raw: object) -> tuple[int, int] | None:
    text = str(raw if raw is not None else "").strip()
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


def _read_chapters_tsv_rows(path: Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for raw in reader:
            row: dict[str, str] = {}
            has_any = False
            for key, value in (raw or {}).items():
                k = str(key or "").strip()
                if not k:
                    continue
                v = str(value or "").strip()
                row[k] = v
                if v:
                    has_any = True
            if has_any:
                rows.append(row)
    return rows


def _chapter_from_tsv_row(archive: str, row: dict[str, str]) -> dict | None:
    lower: dict[str, str] = {}
    for _k, _v in dict(row or {}).items():
        _key = str(_k or "").strip().lower()
        _val = str(_v or "").strip()
        if _key and (_key not in lower or _val):
            lower[_key] = _val
    title = str(lower.get("title") or lower.get("chaptertitle") or lower.get("chapter_title") or "").strip()
    if not title:
        return None

    start_frame = _parse_int_value(lower.get("start_frame"))
    end_frame = _parse_int_value(lower.get("end_frame"))
    start_sec: float | None = None
    end_sec: float | None = None

    if start_frame is None or end_frame is None:
        tb = _parse_timebase_value(lower.get("timebase"))
        start_raw = _parse_int_value(lower.get("start_raw"))
        end_raw = _parse_int_value(lower.get("end_raw"))
        if start_raw is None:
            start_raw = _parse_int_value(lower.get("start"))
        if end_raw is None:
            end_raw = _parse_int_value(lower.get("end"))

        if tb is not None and start_raw is not None and end_raw is not None:
            tb_num, tb_den = tb
            start_sec = float(Fraction(int(start_raw), 1) * Fraction(int(tb_num), int(tb_den)))
            end_sec = float(Fraction(int(end_raw), 1) * Fraction(int(tb_num), int(tb_den)))
            start_frame, end_frame = chapter_frame_bounds(
                {
                    "start_raw": int(start_raw),
                    "end_raw": int(end_raw),
                    "timebase_num": int(tb_num),
                    "timebase_den": int(tb_den),
                },
                fps_num=30000,
                fps_den=1001,
            )
        else:
            if tb is None and start_raw is not None and end_raw is not None:
                start_frame = int(start_raw)
                end_frame = int(end_raw)
            if start_frame is None or end_frame is None:
                start_seconds = _parse_float_value(lower.get("start_seconds"))
                end_seconds = _parse_float_value(lower.get("end_seconds"))
                if start_seconds is None:
                    start_seconds = _parse_float_value(lower.get("start"))
                if end_seconds is None:
                    end_seconds = _parse_float_value(lower.get("end"))
                if start_seconds is not None and end_seconds is not None:
                    start_sec = float(start_seconds)
                    end_sec = float(end_seconds)
                    start_frame = int(round(float(start_seconds) * 30000.0 / 1001.0))
                    end_frame = int(round(float(end_seconds) * 30000.0 / 1001.0))

    if start_frame is None or end_frame is None:
        return None

    start_i, end_i = _normalize_frame_span(int(start_frame), int(end_frame))
    if start_sec is None:
        start_sec = float(start_i) * 1001.0 / 30000.0
    if end_sec is None:
        end_sec = float(end_i) * 1001.0 / 30000.0

    return {
        "title": title,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "start_frame": int(start_i),
        "end_frame": int(end_i),
        "bad_frames": get_bad_frames_for_chapter(str(archive or ""), title),
    }


def load_archive_chapters_tsv(path: Path, archive: str) -> list[dict]:
    rows = _read_chapters_tsv_rows(path)
    out: list[dict] = []
    for row in rows:
        parsed = _chapter_from_tsv_row(archive, row)
        if parsed is None:
            continue
        out.append(parsed)
    return out


def _load_archive_chapters_for_ui(archive: str) -> tuple[list[dict], str]:
    archive_name = str(archive or "").strip()
    if not archive_name:
        return [], ""

    tsv_path = _chapters_tsv_path(archive_name)
    if tsv_path.exists():
        chapters = load_archive_chapters_tsv(tsv_path, archive=archive_name)
        if chapters:
            return chapters, "chapters.tsv"

    ffmetadata_path = _chapters_ffmetadata_path(archive_name)
    if ffmetadata_path.exists():
        chapters = load_archive_chapters(ffmetadata_path)
        if chapters:
            return chapters, "chapters.ffmetadata"

    return [], ""


def _normalize_frame_span(ch_start: int, ch_end: int) -> tuple[int, int]:
    # Half-open chapter range: [start, end)
    start = int(ch_start)
    end = int(ch_end)
    if end < start:
        start, end = end, start
    if end == start:
        end = start + 1
    return start, end


def _env_truthy(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return max(minimum, int(default))
    try:
        val = int(raw)
    except Exception:
        return max(minimum, int(default))
    return max(minimum, val)


def _source_signature_token(source: str | Path | None) -> str:
    if source is None:
        return "nosrc"
    p = Path(source)
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p
    try:
        st = p.stat()
        marker = f"{resolved}|{int(st.st_size)}|{int(st.st_mtime_ns)}"
    except Exception:
        marker = f"{resolved}|missing"
    return hashlib.blake2b(marker.encode("utf-8"), digest_size=8).hexdigest()


def _cleanup_tuner_cache(force: bool = False) -> None:
    global _LAST_CACHE_CLEANUP_TS
    now = time.time()
    interval_sec = _env_int("VHS_TUNER_CACHE_CLEANUP_INTERVAL_SEC", default=300, minimum=30)
    if not force and (now - _LAST_CACHE_CLEANUP_TS) < float(interval_sec):
        return
    _LAST_CACHE_CLEANUP_TS = now

    root = TUNER_CACHE_ROOT
    if not root.exists():
        return

    ttl_days = _env_int("VHS_TUNER_CACHE_TTL_DAYS", default=14, minimum=1)
    max_bytes = _env_int(
        "VHS_TUNER_CACHE_MAX_BYTES",
        default=2 * 1024 * 1024 * 1024,
        minimum=64 * 1024 * 1024,
    )
    cutoff = now - (float(ttl_days) * 86400.0)

    files: list[tuple[float, int, Path]] = []
    total_bytes = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except Exception:
            continue
        mtime = float(st.st_mtime)
        size = int(max(0, st.st_size))
        if mtime < cutoff:
            try:
                p.unlink()
            except Exception:
                pass
            continue
        files.append((mtime, size, p))
        total_bytes += size

    if total_bytes > max_bytes:
        files.sort(key=lambda x: x[0])
        for _mtime, size, p in files:
            if total_bytes <= max_bytes:
                break
            try:
                p.unlink()
                total_bytes -= size
            except Exception:
                continue

    dirs = [p for p in root.rglob("*") if p.is_dir()]
    dirs.sort(key=lambda p: len(p.parts), reverse=True)
    for d in dirs:
        try:
            d.rmdir()
        except Exception:
            continue


def _chapter_extract_cache_path(
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
    debug_overlay: bool,
    source_video: str | Path | None = None,
) -> Path:
    start_i, end_i = _normalize_frame_span(ch_start, ch_end)
    mode = "debug" if bool(debug_overlay) else "clean"
    source_sig = _source_signature_token(source_video)
    key_raw = "|".join(
        [
            str(archive or "").strip(),
            str(chapter_title or "").strip(),
            str(int(start_i)),
            str(int(end_i)),
            str(mode),
            str(source_sig),
        ]
    )
    key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()[:16]
    return TUNER_EXTRACT_DIR / key / "extracted.mkv"


def _probe_video_frame_count(path: Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    cmd = [
        str(FFPROBE_BIN),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "default=noprint_wrappers=1",
        str(p),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception:
        return 0
    counts: dict[str, int] = {}
    for raw in str(out or "").splitlines():
        line = str(raw or "").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        token = str(value or "").strip()
        if not token or token.upper() == "N/A":
            continue
        try:
            parsed = int(token)
        except Exception:
            continue
        if parsed > 0:
            counts[str(key or "").strip()] = int(parsed)
    return int(counts.get("nb_read_frames") or counts.get("nb_frames") or 0)


def _video_frame_count(path: Path) -> int:
    p = Path(path)
    probed = _probe_video_frame_count(p)
    if probed > 0:
        return int(probed)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return 0
    try:
        metadata_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if metadata_count > 0:
            return int(metadata_count)
        decoded_count = 0
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            decoded_count += 1
        return int(decoded_count)
    finally:
        cap.release()


def _ensure_render_chapter_extract(
    *,
    source_video: Path,
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
    debug_overlay: bool,
) -> tuple[Path | None, str]:
    _cleanup_tuner_cache()
    start_i, end_i = _normalize_frame_span(ch_start, ch_end)
    expected_frames = max(1, end_i - start_i)
    out_path = _chapter_extract_cache_path(
        archive=archive,
        chapter_title=chapter_title,
        ch_start=start_i,
        ch_end=end_i,
        debug_overlay=debug_overlay,
        source_video=source_video,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and _video_frame_count(out_path) == expected_frames:
        return out_path, ""

    # The tuner reviews and labels individual frames, so this extract must stay
    # frame-accurate with respect to [ch_start, ch_end). A stream-copy trim is
    # faster but can drift to nearby keyframes and break bad-frame review.
    start_sec = float(start_i) * 1001.0 / 30000.0
    end_sec = float(end_i) * 1001.0 / 30000.0
    cmd = make_frame_accurate_extract_chapter(
        source_video,
        start_sec,
        end_sec,
        out_path,
        start_frame=start_i,
        end_frame=end_i,
        debug_frame_numbers=bool(debug_overlay),
    )
    try:
        proc = subprocess.run(
            [str(x) for x in cmd],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return (
            None,
            "ffmpeg not found: "
            f"{cmd[0]}\nRun 'uv run python scripts/bootstrap_runtime.py' to extract bundled binaries.",
        )
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout or "ffmpeg extraction failed").strip()
    actual_frames = _video_frame_count(out_path)
    if actual_frames != expected_frames:
        actual_note = f", got {actual_frames}" if actual_frames > 0 else ", could not determine actual frame count"
        return None, (
            f"Extracted chapter frame count mismatch for {out_path.name}: expected {expected_frames}{actual_note}"
        )
    return out_path, ""


def slugify(title: str) -> str:
    return re.sub(r"[^\w]+", "_", str(title).strip()).strip("_").lower()


def _chapters_file_path(archive: str) -> Path:
    return METADATA_DIR / str(archive or "").strip() / "chapters.ffmetadata"


def _chapter_bad_overrides(
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
) -> dict[int, str]:
    # Seed per-session overrides from persisted chapter BAD_FRAMES so a chapter
    # reload reflects previously saved decisions in the sampled frame view.
    start, end = _normalize_frame_span(ch_start, ch_end)
    bad_frames = get_bad_frames_for_chapter(str(archive or ""), str(chapter_title or ""), ch_start=start, ch_end=end)
    out: dict[int, str] = {}
    for fid in bad_frames or []:
        try:
            fi = int(fid)
        except Exception:
            continue
        if start <= fi < end:
            out[fi] = "bad"
    return out


def _persist_visible_bad_frames(
    *,
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
    fids: list[int],
    sigs: dict[str, np.ndarray],
    overrides: dict[int, str],
    wc: float,
    wn: float,
    wt: float,
    ww: float,
    tm: str,
    ik: float,
    tv: float,
    bp: float,
    force_all_frames_good: bool = False,
) -> tuple[Path | None, int]:
    if not archive or not chapter_title or not fids or not sigs:
        return None, 0
    cf = _chapters_file_path(archive)
    if not cf.exists():
        return None, 0
    chapters = load_archive_chapters(cf)
    ch = _find_chapter(chapters, chapter_title)
    if not ch:
        return None, 0

    start, end = _normalize_frame_span(ch_start, ch_end)
    existing_global_bad = {int(x) for x in ch.get("bad_frames", []) if start <= int(x) < end}

    scores = combined_score(sigs, wc, wn, wt, ww)
    thr = compute_threshold(scores, tm, ik, tv, bp)

    for fid, sc in zip(fids, scores):
        fid_i = int(fid)
        if not (start <= fid_i < end):
            continue
        if bool(force_all_frames_good):
            is_bad = False
        else:
            ov = overrides.get(fid_i)
            if ov == "bad":
                is_bad = True
            elif ov == "good":
                is_bad = False
            else:
                is_bad = bool(float(sc) >= float(thr))
        if is_bad:
            existing_global_bad.add(int(fid_i))
        else:
            existing_global_bad.discard(int(fid_i))

    out_global = sorted(existing_global_bad)
    out_path = replace_chapter_bad_frames_in_render_settings(
        str(archive or ""),
        start,
        end,
        out_global,
    )
    return out_path, len(out_global)


def persist_bad_frames_for_chapter(
    *,
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
    fids: list[int],
    sigs: dict[str, np.ndarray],
    overrides: dict[int, str],
    wc: float,
    wn: float,
    wt: float,
    ww: float,
    tm: str,
    ik: float,
    tv: float,
    bp: float,
    progress=None,
    force_all_frames_good: bool = False,
) -> tuple[Path | None, int, int, str]:
    _ = progress
    start, end = _normalize_frame_span(ch_start, ch_end)
    sampled_fids = [int(x) for x in (fids or [])]
    sampled_sigs = sigs or {}
    analyzed = len(sampled_fids)
    if analyzed == 0 or not sampled_sigs:
        return None, 0, analyzed, "No sampled frames loaded."

    path, count = _persist_visible_bad_frames(
        archive=str(archive or ""),
        chapter_title=str(chapter_title or ""),
        ch_start=start,
        ch_end=end,
        fids=sampled_fids,
        sigs=sampled_sigs,
        overrides=overrides or {},
        wc=wc,
        wn=wn,
        wt=wt,
        ww=ww,
        tm=tm,
        ik=ik,
        tv=tv,
        bp=bp,
        force_all_frames_good=bool(force_all_frames_good),
    )
    return path, int(count), analyzed, ""


def _signals_cache_path(
    archive: str,
    ch_title: str,
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    frame_read_offset: int = 0,
) -> Path:
    s, e = _normalize_frame_span(start_frame, end_frame)
    source_sig = _source_signature_token(video_path)
    key_raw = "|".join(
        [
            str(int(TUNER_FRAME_CACHE_VERSION)),
            str(archive or "").strip(),
            str(ch_title or "").strip(),
            str(int(s)),
            str(int(e)),
            str(int(frame_read_offset)),
            str(source_sig),
        ]
    )
    key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()[:16]
    return TUNER_FRAME_CACHE_DIR / f"{key}.json.gz"


def load_cached_signals(
    archive: str,
    ch_title: str,
    *,
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    frame_read_offset: int = 0,
) -> tuple[list[int] | None, dict | None, dict[int, str] | None]:
    path = _signals_cache_path(
        archive=archive,
        ch_title=ch_title,
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_read_offset=frame_read_offset,
    )
    if not path.exists():
        return None, None, None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return None, None, None

    try:
        version = int(payload.get("version", -1))
    except Exception:
        return None, None, None
    if version != int(TUNER_FRAME_CACHE_VERSION):
        return None, None, None

    fids_raw = payload.get("fids", [])
    sigs_raw = payload.get("signals", {})
    thumbs_raw = payload.get("thumbs", {})

    if not isinstance(fids_raw, list) or not isinstance(sigs_raw, dict):
        return None, None, None
    try:
        fids = [int(x) for x in fids_raw]
    except Exception:
        return None, None, None

    out_sigs: dict[str, np.ndarray] = {}
    for key in _CACHE_SIGNAL_KEYS:
        vals = sigs_raw.get(key)
        if not isinstance(vals, list) or len(vals) != len(fids):
            return None, None, None
        try:
            out_sigs[key] = np.asarray([float(v) for v in vals], dtype=np.float64)
        except Exception:
            return None, None, None

    thumbs: dict[int, str] = {}
    if isinstance(thumbs_raw, dict):
        for raw_fid, raw_b64 in thumbs_raw.items():
            if not isinstance(raw_b64, str):
                continue
            try:
                fid_i = int(raw_fid)
            except Exception:
                continue
            thumbs[fid_i] = raw_b64

    try:
        path.touch()
    except Exception:
        pass
    return fids, out_sigs, thumbs


def save_cached_signals(
    archive: str,
    ch_title: str,
    *,
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    frame_read_offset: int = 0,
    fids: list[int],
    sigs: dict,
    thumbs_by_fid: dict[int, str] | None = None,
) -> None:
    if not fids or not sigs:
        return

    fids_out = [int(x) for x in fids]
    n = len(fids_out)
    sigs_out: dict[str, list[float]] = {}
    for key in _CACHE_SIGNAL_KEYS:
        arr = np.asarray(sigs.get(key, []), dtype=np.float64)
        if int(arr.shape[0]) != n:
            return
        sigs_out[key] = [float(v) for v in arr.tolist()]

    fid_set = set(fids_out)
    thumbs_out: dict[str, str] = {}
    for raw_fid, raw_b64 in dict(thumbs_by_fid or {}).items():
        try:
            fid_i = int(raw_fid)
        except Exception:
            continue
        if fid_i not in fid_set:
            continue
        b64 = str(raw_b64 or "")
        if b64:
            thumbs_out[str(fid_i)] = b64

    s, e = _normalize_frame_span(start_frame, end_frame)
    path = _signals_cache_path(
        archive=archive,
        ch_title=ch_title,
        video_path=video_path,
        start_frame=s,
        end_frame=e,
        frame_read_offset=frame_read_offset,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": int(TUNER_FRAME_CACHE_VERSION),
        "archive": str(archive or ""),
        "chapter": str(ch_title or ""),
        "start_frame": int(s),
        "end_frame": int(e),
        "frame_read_offset": int(frame_read_offset),
        "source_sig": _source_signature_token(video_path),
        "updated_at": float(time.time()),
        "fids": fids_out,
        "signals": sigs_out,
        "thumbs": thumbs_out,
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except Exception:
            pass
        return
    _cleanup_tuner_cache()


def _compute_signals(bgr: np.ndarray, crop: int = 50) -> tuple[float, float, float, float]:
    h, w = bgr.shape[:2]
    y0 = min(crop, max(0, h - 1))
    y1 = max(y0 + 1, h - crop)
    x0 = min(crop, max(0, w - 1))
    x1 = max(x0 + 1, w - crop)
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        roi = bgr
    s = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    chroma_loss = 1.0 - float(np.mean(s) / 255.0)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    row_vars = np.var(gray, axis=1)
    mean_var = float(np.mean(row_vars))
    noise = float(np.std(row_vars) / mean_var) if mean_var > 1e-6 else 0.0
    tear = float(np.percentile(np.abs(gray[1:] - gray[:-1]).mean(axis=1), 95)) if gray.shape[0] > 1 else 0.0
    row_sums = gray.sum(axis=1)
    cols_idx = np.arange(gray.shape[1], dtype=np.float32)
    row_com = (gray @ cols_idx) / np.maximum(row_sums, 1e-6)
    wave = (
        float(np.std(row_com - np.convolve(row_com, np.ones(5) / 5, mode="same")))
        if row_com.shape[0] >= 5
        else float(np.std(row_com))
    )
    return chroma_loss, noise, tear, wave


def _bgr_to_jpeg_b64(bgr: np.ndarray, width: int = 160) -> str:
    h, w = bgr.shape[:2]
    thumb = cv2.resize(bgr, (width, int(width * h / max(w, 1))), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def extract_frames(
    video_path: str,
    start: int,
    end: int,
    n: int,
    archive: str,
    ch_title: str,
    frame_ids: list[int] | None = None,
    include_thumbs: bool = True,
    frame_read_offset: int = 0,
    progress=None,
    should_cancel=None,
    frame_callback=None,
) -> tuple[list[int] | None, list[str] | None, dict | None, str]:
    start_i, end_i = _normalize_frame_span(start, end)
    if frame_ids is None:
        target_n = max(1, min(int(n), max(1, end_i - start_i)))
        frame_ids = np.linspace(start_i, end_i - 1, target_n, dtype=int).tolist()
    else:
        frame_ids = [int(x) for x in frame_ids if start_i <= int(x) < end_i]
        frame_ids = sorted(set(frame_ids))
        if not frame_ids:
            target_n = max(1, min(int(n), max(1, end_i - start_i)))
            frame_ids = np.linspace(start_i, end_i - 1, target_n, dtype=int).tolist()
    assert frame_ids is not None
    frame_set = set(frame_ids)

    cached_fids, cached_sigs, cached_thumbs = load_cached_signals(
        archive,
        ch_title,
        video_path=video_path,
        start_frame=start_i,
        end_frame=end_i,
        frame_read_offset=frame_read_offset,
    )
    cached_lookup: dict[int, dict[str, float]] = {}
    if cached_fids and cached_sigs:
        for i, fid in enumerate(cached_fids):
            cached_lookup[fid] = {k: float(v[i]) for k, v in cached_sigs.items()}
    thumb_lookup: dict[int, str] = dict(cached_thumbs or {})

    cap: cv2.VideoCapture | None = None

    frames_b64: list[str] = []
    chroma_s, noise_s, tear_s, wave_s = [], [], [], []

    read_offset = int(frame_read_offset)
    n_total = max(1, len(frame_ids))
    prev_read_fid: int | None = None
    for idx, fid in enumerate(frame_ids):
        if callable(should_cancel) and bool(should_cancel()):
            if cap is not None:
                cap.release()
            return None, None, None, "Load cancelled."
        read_fid = int(fid) - read_offset
        if progress is not None:
            progress(idx / n_total, desc=f"Frame {fid}...")

        c = cached_lookup.get(int(fid))
        cached_thumb = thumb_lookup.get(int(fid), "")
        need_decode = c is None or (include_thumbs and not cached_thumb)

        bgr = None
        if need_decode:
            if read_fid < 0:
                bgr = np.zeros((240, 320, 3), dtype=np.uint8)
            else:
                if cap is None:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        return None, None, None, f"Cannot open video: {video_path}"
                # Skip seek when frames are sequential — cap is already at the right position
                if prev_read_fid is None or read_fid != prev_read_fid + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(read_fid))
                ok, read_bgr = cap.read()
                prev_read_fid = read_fid
                if not ok or read_bgr is None:
                    bgr = np.zeros((240, 320, 3), dtype=np.uint8)
                else:
                    bgr = read_bgr

        if include_thumbs:
            if cached_thumb:
                frame_thumb = cached_thumb
            else:
                frame_thumb = _bgr_to_jpeg_b64(bgr if bgr is not None else np.zeros((240, 320, 3), dtype=np.uint8))
                thumb_lookup[int(fid)] = frame_thumb
            frames_b64.append(frame_thumb)
        else:
            frame_thumb = ""

        if c is not None:
            ch = float(c["chroma"])
            no = float(c["noise"])
            te = float(c["tear"])
            wa = float(c["wave"])
            chroma_s.append(ch)
            noise_s.append(no)
            tear_s.append(te)
            wave_s.append(wa)
        else:
            compute_bgr = bgr if bgr is not None else np.zeros((240, 320, 3), dtype=np.uint8)
            ch, no, te, wa = _compute_signals(compute_bgr)
            chroma_s.append(ch)
            noise_s.append(no)
            tear_s.append(te)
            wave_s.append(wa)
        if callable(frame_callback):
            frame_callback(
                int(fid),
                frame_thumb,
                float(ch),
                float(no),
                float(te),
                float(wa),
                idx + 1,
                len(frame_ids),
            )

    if cap is not None:
        cap.release()

    sigs: dict[str, np.ndarray] = {
        "chroma": np.array(chroma_s, dtype=np.float64),
        "noise": np.array(noise_s, dtype=np.float64),
        "tear": np.array(tear_s, dtype=np.float64),
        "wave": np.array(wave_s, dtype=np.float64),
    }

    # Merge into persistent cache
    all_fids_l: list[int] = list(frame_ids)
    all_sigs_l: dict[str, list[float]] = {k: list(v) for k, v in sigs.items()}
    if cached_fids and cached_sigs:
        for i, fid in enumerate(cached_fids):
            if fid not in frame_set:
                all_fids_l.append(fid)
                for k, arr in cached_sigs.items():
                    all_sigs_l[k].append(float(arr[i]))
    order = list(np.argsort(all_fids_l))
    sorted_fids = [all_fids_l[i] for i in order]
    sorted_sigs = {k: np.array([v[i] for i in order]) for k, v in all_sigs_l.items()}
    save_cached_signals(
        archive,
        ch_title,
        video_path=video_path,
        start_frame=start_i,
        end_frame=end_i,
        frame_read_offset=frame_read_offset,
        fids=sorted_fids,
        sigs=sorted_sigs,
        thumbs_by_fid=thumb_lookup if include_thumbs else None,
    )

    return frame_ids, frames_b64, sigs, ""


# ===============================================================================
# Scoring / thresholding (shared in common.py)
# ===============================================================================


def toggle_frame_override(
    fid: int,
    fids: list[int],
    sigs: dict[str, np.ndarray],
    overrides: dict[int, str],
    wc: float,
    wn: float,
    wt: float,
    ww: float,
    tm: str,
    ik: float,
    tv: float,
    bp: float,
) -> dict[int, str]:
    out = dict(overrides)
    # 3-state manual override:
    # - no override + auto-good -> force bad
    # - no override + auto-bad  -> force good
    # - forced bad/good         -> clear override (back to auto)
    if int(fid) not in out:
        scores = combined_score(sigs, wc, wn, wt, ww)
        thr = compute_threshold(scores, tm, ik, tv, bp)
        pos = {int(f): i for i, f in enumerate(fids)}.get(int(fid))
        auto_bad = bool(pos is not None and float(scores[pos]) >= float(thr))
        out[int(fid)] = "good" if auto_bad else "bad"
    else:
        del out[int(fid)]
    return out


def set_frame_override_mode(
    fid: int,
    fids: list[int],
    sigs: dict[str, np.ndarray],
    overrides: dict[int, str],
    wc: float,
    wn: float,
    wt: float,
    ww: float,
    tm: str,
    ik: float,
    tv: float,
    bp: float,
    mode: str = "toggle",
) -> dict[int, str]:
    out = dict(overrides or {})
    mode_n = str(mode or "toggle").strip().lower()
    if mode_n == "bad":
        out[int(fid)] = "bad"
        return out
    if mode_n == "good":
        out[int(fid)] = "good"
        return out
    if mode_n in {"clear", "auto"}:
        out.pop(int(fid), None)
        return out
    return toggle_frame_override(
        fid=fid,
        fids=fids,
        sigs=sigs,
        overrides=out,
        wc=wc,
        wn=wn,
        wt=wt,
        ww=ww,
        tm=tm,
        ik=ik,
        tv=tv,
        bp=bp,
    )


def _parse_click_payload(raw_click: str) -> tuple[int, int]:
    text = str(raw_click or "").strip()
    if not text:
        raise ValueError("empty click payload")
    parts = text.split(":")
    fid = int(parts[0])
    # Use server receive time for dedupe; client clocks can differ.
    ts = int(time.time() * 1000)
    return fid, ts


def _should_dedupe_click(
    *,
    fid: int,
    ts: int,
    last_click_event: dict | None,
    window_ms: int = 220,
) -> bool:
    if not isinstance(last_click_event, dict):
        return False
    try:
        last_fid = int(last_click_event.get("fid", -1))
        last_ts = int(last_click_event.get("ts", -1))
    except Exception:
        return False
    if fid != last_fid:
        return False
    dt = int(ts) - int(last_ts)
    return 0 <= dt <= max(0, int(window_ms))


def apply_manual_click_override(
    *,
    raw_click: str,
    fids: list[int],
    sigs: dict[str, np.ndarray],
    overrides: dict[int, str],
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
    wc: float,
    wn: float,
    wt: float,
    ww: float,
    tm: str,
    ik: float,
    tv: float,
    bp: float,
    mark_mode: str = "toggle",
    last_click_event: dict | None = None,
) -> tuple[dict[int, str], dict[str, int], str]:
    current = dict(overrides or {})
    try:
        fid, ts = _parse_click_payload(raw_click)
    except Exception:
        return (
            current,
            dict(last_click_event or {}),
            f"ignored: invalid payload '{raw_click}'",
        )

    if fid not in {int(x) for x in fids}:
        return (
            current,
            dict(last_click_event or {}),
            f"ignored: frame {fid} not in sampled set",
        )

    if _should_dedupe_click(fid=fid, ts=ts, last_click_event=last_click_event):
        return (
            current,
            dict(last_click_event or {}),
            (f"ignored: duplicate click fid={fid} ts={ts}"),
        )

    before = current.get(fid)
    new_ov = set_frame_override_mode(
        fid=fid,
        fids=fids,
        sigs=sigs,
        overrides=current,
        wc=wc,
        wn=wn,
        wt=wt,
        ww=ww,
        tm=tm,
        ik=ik,
        tv=tv,
        bp=bp,
        mode=mark_mode,
    )

    after = new_ov.get(fid)
    srv_dbg = (
        f"payload={raw_click} fid={fid} ts={ts} mode={mark_mode} before={before} after={after} "
        "persisted=False (explicit save required)"
    )
    print(f"[vhs_tuner] {srv_dbg}")
    return new_ov, {"fid": int(fid), "ts": int(ts)}, srv_dbg


# ===============================================================================
# AI-Agent Analysis Helpers
# ===============================================================================


def suggest_iqr_k(scores: np.ndarray) -> dict:
    """
    Suggest an optimal IQR k value by finding the natural gap between good/bad frames.

    Iterates k from 0.5 to 8.0, measuring the score gap between the lowest bad
    frame and the highest good frame at each threshold. A wide gap means the
    threshold sits in a clean valley; gap_score = gap_width * log2(1 + bad_count)
    rewards gaps that also capture a meaningful number of bad frames.

    Returns suggested_k, a confidence in [0, 1], and the top 5 k candidates.
    """
    import math

    v = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    v = v[np.isfinite(v)]
    if v.size < 4:
        return {
            "suggested_k": 3.5,
            "confidence": 0.0,
            "threshold_at_k": float(v[-1]) if v.size else 0.0,
            "bad_count_at_k": 0,
            "percent_bad_at_k": 0.0,
            "candidates": [],
        }

    q1 = float(np.percentile(v, 25))
    q3 = float(np.percentile(v, 75))
    iqr = q3 - q1
    if iqr < 1e-9:
        return {
            "suggested_k": 3.5,
            "confidence": 0.0,
            "threshold_at_k": float(q3),
            "bad_count_at_k": 0,
            "percent_bad_at_k": 0.0,
            "candidates": [],
        }

    candidates = []
    for k_int in range(2, 33):  # k = 0.5 to 8.0 in steps of 0.25
        k = k_int * 0.25
        thr = q3 + k * iqr
        above = v[v >= thr]
        below = v[v < thr]
        bad_count = len(above)
        if bad_count == 0 or len(below) == 0:
            gap = 0.0
        else:
            gap = max(0.0, float(above[0]) - float(below[-1]))
        gap_score = gap * math.log2(1.0 + bad_count)
        candidates.append(
            {
                "k": round(k, 2),
                "threshold": round(float(thr), 4),
                "bad_count": int(bad_count),
                "gap_width": round(gap, 4),
                "gap_score": round(gap_score, 4),
            }
        )

    candidates.sort(key=lambda x: x["gap_score"], reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else best
    confidence = 1.0 - (second["gap_score"] / (best["gap_score"] + 1e-9))
    confidence = max(0.0, min(1.0, round(confidence, 3)))

    return {
        "suggested_k": best["k"],
        "confidence": confidence,
        "threshold_at_k": best["threshold"],
        "bad_count_at_k": best["bad_count"],
        "percent_bad_at_k": round(100.0 * best["bad_count"] / len(v), 2),
        "candidates": candidates[:5],
    }


def find_spike_regions(
    fids: list[int],
    scores: np.ndarray,
    threshold: float,
    context_frames: int = 8,
) -> list[dict]:
    """
    Find contiguous windows around frames that exceed the score threshold.

    Each bad frame is expanded by context_frames on each side; overlapping
    windows are merged. Returns regions sorted by start_fid, each with peak
    score, peak fid, and count of bad frames within the region.
    """
    if not fids:
        return []

    fids_arr = np.asarray(fids, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    bad_indices = list(np.where(scores_arr >= float(threshold))[0])

    if not bad_indices:
        return []

    n = len(fids)
    include: set[int] = set()
    for idx in bad_indices:
        lo = max(0, int(idx) - context_frames)
        hi = min(n - 1, int(idx) + context_frames)
        include.update(range(lo, hi + 1))

    sorted_indices = sorted(include)
    raw_regions: list[tuple[int, int]] = []
    region_start = sorted_indices[0]
    region_end = sorted_indices[0]
    for idx in sorted_indices[1:]:
        if idx == region_end + 1:
            region_end = idx
        else:
            raw_regions.append((region_start, region_end))
            region_start = idx
            region_end = idx
    raw_regions.append((region_start, region_end))

    result = []
    for i, (lo_idx, hi_idx) in enumerate(raw_regions):
        region_fids = fids_arr[lo_idx : hi_idx + 1]
        region_scores = scores_arr[lo_idx : hi_idx + 1]
        bad_in_region = int(np.sum(region_scores >= float(threshold)))
        peak_local = int(np.argmax(region_scores))
        result.append(
            {
                "region_index": i,
                "start_fid": int(region_fids[0]),
                "end_fid": int(region_fids[-1]),
                "frame_count": int(len(region_fids)),
                "peak_score": round(float(region_scores[peak_local]), 4),
                "peak_fid": int(region_fids[peak_local]),
                "bad_frame_count": bad_in_region,
            }
        )

    return result


def estimate_gamma_from_frames(b64_frames: list[str]) -> dict:
    """
    Estimate optimal gamma from base64-encoded JPEG thumbnails.

    Analyzes the LAB L-channel median luminance across sampled frames and
    computes the gamma needed to bring median luminance to a target of 0.45
    (normalized). Returns suggested gamma clamped to [0.3, 3.0].
    """
    import math as _math
    import base64 as _base64

    TARGET_NORMALIZED = 0.45
    luminances: list[float] = []
    for b64 in b64_frames:
        try:
            raw_b64 = b64.split(",", 1)[1] if b64.startswith("data:") else b64
            raw = _base64.b64decode(raw_b64)
            buf = np.frombuffer(raw, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            mean_l = float(np.mean(lab[:, :, 0].astype(np.float32)))
            luminances.append(mean_l)
        except Exception:
            continue

    if not luminances:
        return {
            "suggested_gamma": 1.0,
            "median_luminance_normalized": 0.0,
            "sample_count": 0,
        }

    median_l = float(np.median(luminances))
    median_norm = median_l / 255.0

    if median_norm <= 0.01 or median_norm >= 0.99:
        suggested = 1.0
    else:
        try:
            suggested = _math.log(TARGET_NORMALIZED) / _math.log(median_norm)
        except (ValueError, ZeroDivisionError):
            suggested = 1.0
    suggested = max(0.3, min(3.0, round(suggested, 2)))

    return {
        "suggested_gamma": suggested,
        "median_luminance_normalized": round(median_norm, 3),
        "sample_count": len(luminances),
    }


# ===============================================================================
# SVG Sparklines  - timeline charts with horizontal red cut line
# ===============================================================================


def _sparkline_svg(
    values: np.ndarray,
    threshold: float | None = None,
    label: str = "",
    height: int = 38,
    line_color: str = "#27a85a",
) -> str:
    """
    Timeline sparkline: X axis = frame index, Y axis = signal value.
    An optional horizontal red line marks the threshold cut.
    Keeps a compact sparkline width while still shrinking to fit narrow layouts.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    SVG_W = 200  # viewBox width - browser scales to container

    if v.size == 0:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {height}" '
            f'style="background:#0a0a0a;display:block;width:220px;max-width:100%;border-radius:2px;margin-bottom:3px">'
            f'<text x="4" y="14" font-family="Courier New" font-size="9" fill="#444">{label}</text>'
            f"</svg>"
        )

    vmin = float(v.min())
    vmax = float(v.max())
    vrange = (vmax - vmin) or 1.0
    n = len(v)
    PAD = 3  # top padding so high points aren't clipped

    def _x(i: int) -> float:
        return i / max(n - 1, 1) * SVG_W

    def _y(val: float) -> float:
        return PAD + (height - PAD) * (1.0 - (val - vmin) / vrange)

    pts = " ".join(f"{_x(i):.1f},{_y(val):.1f}" for i, val in enumerate(v))

    # Filled area under the line
    area_pts = f"0,{height} {pts} {SVG_W},{height}"

    # Threshold line + right-edge triangle marker
    tline = ""
    if threshold is not None:
        ty = _y(threshold)
        ty = max(0.0, min(float(height), ty))
        tline = (
            f'<line x1="0" y1="{ty:.1f}" x2="{SVG_W}" y2="{ty:.1f}" '
            f'stroke="#e03030" stroke-width="1.8" opacity="0.95"/>'
            f'<polygon points="{SVG_W},{ty:.1f} {SVG_W - 6},{ty - 4:.1f} {SVG_W - 6},{ty + 4:.1f}" '
            f'fill="#e03030" opacity="0.9"/>'
        )

    lbl = f'<text x="3" y="{height - 3}" font-family="Courier New" font-size="8" fill="#555">{label}</text>'

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {height}" '
        f'style="background:#0a0a0a;display:block;width:220px;max-width:100%;border-radius:2px;margin-bottom:3px">'
        f'<polygon points="{area_pts}" fill="{line_color}" opacity="0.12"/>'
        f'<polyline points="{pts}" fill="none" stroke="{line_color}" stroke-width="1.3" opacity="0.85"/>'
        f"{tline}{lbl}"
        f"</svg>"
    )


def build_sparklines_html(
    sigs: dict,
    scores: np.ndarray,
    threshold: float,
    wc: float,
    wn: float,
    wt: float,
    ww: float,
) -> tuple[str, str, str, str, str]:
    """
    Returns (spark_chroma, spark_noise, spark_tear, spark_wave, spark_score).

    Signal sparklines show raw values over the sampled frames.
    The line opacity tracks the current weight (dim when weight is low).
    The composite score sparkline carries the red threshold line.
    """

    def _col(w: float) -> str:
        alpha = 0.25 + 0.75 * min(1.0, w / 0.5)
        return f"rgba(39,168,90,{alpha:.2f})"

    def _unit(v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return arr
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - lo) / (hi - lo)

    sc_chroma = _sparkline_svg(
        _unit(sigs.get("chroma", np.array([]))),
        wc,
        f"chroma  w={wc:.2f}",
        height=24,
        line_color=_col(wc),
    )
    sc_noise = _sparkline_svg(
        _unit(sigs.get("noise", np.array([]))),
        wn,
        f"noise   w={wn:.2f}",
        height=24,
        line_color=_col(wn),
    )
    sc_tear = _sparkline_svg(
        _unit(sigs.get("tear", np.array([]))),
        wt,
        f"tear    w={wt:.2f}",
        height=24,
        line_color=_col(wt),
    )
    sc_wave = _sparkline_svg(
        _unit(sigs.get("wave", np.array([]))),
        ww,
        f"wave    w={ww:.2f}",
        height=24,
        line_color=_col(ww),
    )
    sc_score = _sparkline_svg(scores, threshold, "composite score", height=32, line_color="#5599dd")

    return sc_chroma, sc_noise, sc_tear, sc_wave, sc_score


# Chapter list helpers
def _get_archives() -> list[str]:
    names: set[str] = set()
    if ARCHIVE_DIR.exists():
        names.update(p.stem for p in ARCHIVE_DIR.glob("*.mkv"))
    if METADATA_DIR.exists():
        names.update(p.name for p in METADATA_DIR.iterdir() if p.is_dir())
    return sorted(names)


def _resolve_archive_video(archive: str) -> Path | None:
    proxy = ARCHIVE_DIR / f"{archive}_proxy.mp4"
    mkv = ARCHIVE_DIR / f"{archive}.mkv"
    return proxy if proxy.exists() else mkv if mkv.exists() else None


CHAPTER_SELECT_LABEL = "-- select chapter --"
CHAPTER_MISSING_LABEL = "-- no chapters metadata found --"


def _get_chapter_titles(archive: str, chapters: list[dict] | None = None) -> list[str]:
    if not archive:
        return [CHAPTER_SELECT_LABEL]
    chapter_rows = list(chapters or [])
    if not chapter_rows:
        chapter_rows, _source = _load_archive_chapters_for_ui(archive)
    if not chapter_rows:
        return [CHAPTER_MISSING_LABEL]
    return [CHAPTER_SELECT_LABEL] + [str(ch.get("title", "")) for ch in chapter_rows if str(ch.get("title", ""))]


def _find_chapter(chapters: list[dict], title: str) -> dict | None:
    return next((c for c in chapters if c["title"] == title), None)


def _fmt_hms(total_sec: float) -> str:
    sec_i = max(0, int(round(float(total_sec))))
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _chapter_frame_count(ch: dict) -> int:
    start_i, end_i = _normalize_frame_span(
        int(ch.get("start_frame", 0)),
        int(ch.get("end_frame", 0)),
    )
    return max(0, end_i - start_i)


def _chapter_bad_count(ch: dict) -> int:
    start_i, end_i = _normalize_frame_span(
        int(ch.get("start_frame", 0)),
        int(ch.get("end_frame", 0)),
    )
    bad = [int(x) for x in (ch.get("bad_frames", []) or []) if start_i <= int(x) < end_i]
    return len(bad)


def _chapter_rows(chapters: list[dict]) -> tuple[list[list], list[list]]:
    rows: list[list] = []
    compact_rows: list[list] = []
    for idx, ch in enumerate(chapters):
        title = str(ch.get("title", "Untitled"))
        dur = max(0.0, float(ch.get("end_sec", 0.0)) - float(ch.get("start_sec", 0.0)))
        frame_count = _chapter_frame_count(ch)
        bad_count = _chapter_bad_count(ch)
        rows.append([idx + 1, title, _fmt_hms(dur), frame_count, bad_count])
        compact_rows.append([idx + 1, bad_count])
    return rows, compact_rows


def _chapter_details_md(ch: dict | None) -> str:
    if not ch:
        return "`Select a chapter from the left rail.`"
    title = str(ch.get("title", "Untitled"))
    dur = max(0.0, float(ch.get("end_sec", 0.0)) - float(ch.get("start_sec", 0.0)))
    frame_count = _chapter_frame_count(ch)
    bad_count = _chapter_bad_count(ch)
    start_f = int(ch.get("start_frame", 0))
    end_f = int(ch.get("end_frame", 0))
    return (
        f"**{title}**\n\n"
        f"Duration: `{_fmt_hms(dur)}`  |  "
        f"Frames: `{frame_count}`  |  "
        f"BAD already: `{bad_count}`\n\n"
        f"Frame span: `{start_f}` - `{end_f}` (end exclusive)"
    )


def build_archive_state(
    archive: str,
    selected_title: str | None = None,
) -> dict:
    chapters, chapter_source = _load_archive_chapters_for_ui(archive)
    titles = _get_chapter_titles(archive, chapters=chapters)
    chapter_titles = [str(ch.get("title", "")) for ch in chapters if str(ch.get("title", ""))]
    chapter_rows, compact_rows = _chapter_rows(chapters)

    if selected_title and str(selected_title) in chapter_titles:
        chapter_value = str(selected_title)
    else:
        chapter_value = chapter_titles[0] if chapter_titles else (titles[0] if titles else CHAPTER_SELECT_LABEL)

    picked = _find_chapter(chapters, chapter_value)
    start_frame = int(picked["start_frame"]) if picked else None
    end_frame = int(picked["end_frame"]) if picked else None
    details = _chapter_details_md(picked)

    tsv_exists = bool(archive and _chapters_tsv_path(archive).exists())
    ffmeta_exists = bool(archive and _chapters_ffmetadata_path(archive).exists())
    if chapter_titles:
        status = ""
    elif chapter_source == "chapters.tsv" or tsv_exists:
        status = "`No chapters available in chapters.tsv for selected archive.`"
    elif chapter_source == "chapters.ffmetadata" or ffmeta_exists:
        status = "`No chapters available in chapters.ffmetadata for selected archive.`"
    else:
        status = "`No chapters metadata found for selected archive.`"

    return {
        "titles": titles,
        "chapters": chapters,
        "chapter_titles": chapter_titles,
        "chapter_rows": chapter_rows,
        "compact_rows": compact_rows,
        "chapter_value": chapter_value,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "details": details,
        "status": status,
    }
