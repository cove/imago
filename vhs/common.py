#
# Common configuration and utility functions for all scripts:
# - Defines platform-specific binary paths (FFmpeg, mediainfo, Whisper, legacy b3sum).
# - Provides SHA3-256 checksum helpers for manifests and verification.
# - Sets up project directories for archives, videos, clips, and metadata.
# - Provides safe filename sanitization, HMS formatting, and subprocess wrappers.
# - Reads FFmetadata chapters and parses them into Python dicts.
# - Checks if chapter files are done based on size and existence.
# - Measures media duration via ffprobe.
#
import json
import os, shutil, subprocess, sys
import hashlib
from dataclasses import replace as dataclass_replace
from pathlib import Path

# ---------------------------------------------------------
# Base Paths
# ---------------------------------------------------------

BASE = Path(__file__).parent.resolve()
FFMPEG_DIR = None
MODES_DIR = BASE.parent / "modes" / "vhs"
WHISPER_MODEL_DIR = MODES_DIR / "WhisperModel"


def _resolve_command(cmd_name, bundled_path=None):
    """
    Resolve an executable command, preferring a bundled path, then PATH lookup.
    Returns (command, parent_dir_or_none).
    """
    if bundled_path is not None:
        p = Path(bundled_path)
        if p.exists():
            p = p.resolve()
            return p, p.parent

    found = shutil.which(cmd_name)
    if found:
        p = Path(found)
        return p, p.parent

    # Keep a simple command token fallback so subprocess can still try PATH.
    return Path(cmd_name), None


def _command_exists(cmd):
    cmd_text = str(cmd)
    p = Path(cmd_text)
    if p.is_absolute() or "/" in cmd_text or "\\" in cmd_text:
        return p.exists()
    return shutil.which(cmd_text) is not None


if sys.platform == "win32":
    FFMPEG_DIR = BASE / "software" / "Windows" / "FFmpeg-QTGMC Easy 2025.01.11"
    FFMPEG_BIN = FFMPEG_DIR / "ffmpeg.exe"
    FFPROBE_BIN = FFMPEG_DIR / "ffprobe.exe"
    B3SUM_BIN = BASE / "bin" / "b3sum_windows_x64_bin.exe"
    MEDIAINFO_BIN = BASE / "software" / "Windows" / "MediaInfo" / "MediaInfo.exe"
elif sys.platform == "darwin":
    FFMPEG_DIR = BASE / "bin"
    FFMPEG_BIN = FFMPEG_DIR / "ffmpeg-8.0.1.darwin.arm64"
    FFPROBE_BIN = FFMPEG_DIR / "ffprobe-8.0.1.darwin.arm64"
    B3SUM_BIN = BASE / "bin" / "b3sum"
    MEDIAINFO_BIN = "mediainfo"
elif sys.platform.startswith("linux"):
    ffmpeg_override = os.getenv("FFMPEG_BIN")
    ffprobe_override = os.getenv("FFPROBE_BIN")
    b3sum_override = os.getenv("B3SUM_BIN")
    mediainfo_override = os.getenv("MEDIAINFO_BIN")

    if ffmpeg_override:
        FFMPEG_BIN = Path(ffmpeg_override)
        ffmpeg_dir = (
            Path(ffmpeg_override).parent
            if (Path(ffmpeg_override).is_absolute() or "/" in ffmpeg_override or "\\" in ffmpeg_override)
            else None
        )
    else:
        FFMPEG_BIN, ffmpeg_dir = _resolve_command("ffmpeg", BASE / "bin" / "ffmpeg")

    if ffprobe_override:
        FFPROBE_BIN = Path(ffprobe_override)
        ffprobe_dir = (
            Path(ffprobe_override).parent
            if (Path(ffprobe_override).is_absolute() or "/" in ffprobe_override or "\\" in ffprobe_override)
            else None
        )
    else:
        FFPROBE_BIN, ffprobe_dir = _resolve_command("ffprobe", BASE / "bin" / "ffprobe")

    if b3sum_override:
        B3SUM_BIN = Path(b3sum_override)
    else:
        B3SUM_BIN, _ = _resolve_command("b3sum", BASE / "bin" / "b3sum")

    if mediainfo_override:
        mediainfo_cmd = Path(mediainfo_override)
    else:
        mediainfo_cmd, _ = _resolve_command("mediainfo")

    MEDIAINFO_BIN = mediainfo_cmd
    # Prefer ffmpeg folder for PATH augmentation; fall back to ffprobe folder.
    FFMPEG_DIR = ffmpeg_dir or ffprobe_dir
else:
    raise Exception(f"Unsupported platform: {sys.platform}")

WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Project Directories (shared between scripts)
# ---------------------------------------------------------
def store_for(archive_name: str) -> str:
    """Return the store name — the first underscore-delimited segment of the archive name."""
    return archive_name.split("_")[0]


if os.getenv("TEST_ENV") == "1":
    _base_dir = BASE / "test"
    METADATA_DIR = _base_dir / "metadata"
    VIDEO_DATA_DIR = _base_dir / "video_data"
    # Keep flat dirs so existing test fixtures (empty dirs) keep working unchanged.
    ARCHIVE_DIR = _base_dir / "Archive"
    VIDEOS_DIR = _base_dir / "Videos"
    CLIPS_DIR = _base_dir / "Clips"
    DRIVE_DIR = _base_dir.resolve()
else:
    _base_dir = BASE.parent.parent
    METADATA_DIR = BASE / "metadata"
    VIDEO_DATA_DIR = _base_dir / "video_data"
    ARCHIVE_DIR = VIDEOS_DIR = CLIPS_DIR = None
    DRIVE_DIR = VIDEO_DATA_DIR


def archive_dir_for(archive_name: str) -> Path:
    return VIDEO_DATA_DIR / store_for(archive_name) / "Archive"


def videos_dir_for(archive_name: str) -> Path:
    return VIDEO_DATA_DIR / store_for(archive_name) / "Videos"


def clips_dir_for(archive_name: str) -> Path:
    return VIDEO_DATA_DIR / store_for(archive_name) / "Clips"


def all_store_archive_dirs() -> list[Path]:
    """Return sorted list of Archive/ subdirs for every store under video_data/."""
    if not VIDEO_DATA_DIR.exists():
        return []
    return sorted(
        d / "Archive"
        for d in VIDEO_DATA_DIR.iterdir()
        if d.is_dir() and (d / "Archive").is_dir()
    )


def archive_checksum_file_for(archive_name: str) -> Path:
    return archive_dir_for(archive_name) / "SHA256SUMS"


def drive_checksum_file_for(archive_name: str) -> Path:
    return archive_dir_for(archive_name) / "SHA256SUMS_DRIVE"


QTGMC_DIR = FFMPEG_DIR
DRIVE_CHECKSUM_FILE = VIDEO_DATA_DIR / "SHA256SUMS_DRIVE"

# Add FFmpeg binaries early to PATH so all scripts inherit it
if FFMPEG_DIR:
    os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------
# Shared FFmpeg Settings
# ---------------------------------------------------------


def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))


def format_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run(cmd, cwd=None):
    print("Command: " + " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def make_frame_accurate_extract_chapter(
    src,
    start,
    end,
    dest,
    start_frame=None,
    end_frame=None,
    debug_frame_numbers=False,
    audio_offset_seconds=0.0,
):
    """
    Frame-exact chapter extraction command builder.
    Uses select+setpts for exact frame slicing and optional drawtext overlay.

    Important: the VHS tuner relies on this path being frame-accurate. Do not
    replace it with a faster `-ss/-to -c copy` style trim unless the caller is
    explicitly opting into non-frame-accurate behavior.
    """
    if start_frame is None or end_frame is None:
        raise ValueError("make_frame_accurate_extract_chapter requires start_frame and end_frame.")
    s_frame = int(start_frame)
    e_frame = int(end_frame)
    if e_frame <= s_frame:
        e_frame = s_frame + 1

    vf_filters = [
        f"select='between(n\\,{s_frame}\\,{e_frame - 1})'",
        "setpts=N/FRAME_RATE/TB",
    ]
    if bool(debug_frame_numbers):
        local_label = "%{eif\\:n\\:d}"
        global_label = f"%{{eif\\:n+{s_frame}\\:d}}"
        font_expr = ""
        win_font = Path("C:/Windows/Fonts/consola.ttf")
        if win_font.exists():
            font_expr = "fontfile='C\\:/Windows/Fonts/consola.ttf'"
        vf_filters.append(
            "drawtext="
            + f"text='local={local_label} global={global_label}'"
            + (f":{font_expr}" if font_expr else "")
            + ":x=16:y=16:fontsize=24:"
            + "fontcolor=white:box=1:boxcolor=black@0.55:borderw=2"
        )
    vf_select = ",".join(vf_filters)

    offset = float(audio_offset_seconds or 0.0)
    chapter_dur = max(0.001, float(end) - float(start))
    audio_start_raw = float(start) + offset
    audio_end_raw = float(end) + offset
    silence_prepend_sec = max(0.0, -audio_start_raw)  # > 0 only when offset pushes start < 0
    audio_start_clamped = max(0.0, audio_start_raw)
    # Build audio filter: trim → optional silence prepend → pad end to chapter duration
    af_parts = [
        f"atrim=start={audio_start_clamped:.6f}:end={max(audio_start_clamped + 0.001, audio_end_raw):.6f}",
        "asetpts=PTS-STARTPTS",
    ]
    if silence_prepend_sec > 1e-4:
        af_parts.append(f"adelay={silence_prepend_sec * 1000:.1f}:all=1")
    af_parts.append(f"apad=whole_dur={chapter_dur:.6f}")
    af_trim = ",".join(af_parts)
    return [
        FFMPEG_BIN,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(src),
        "-vf",
        vf_select,
        "-af",
        af_trim,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-fps_mode:v:0",
        "passthrough",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-coder",
        "1",
        "-context",
        "1",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "48000",
        "-ac",
        "1",
        "-fflags",
        "+genpts",
        "-start_at_zero",
        "-avoid_negative_ts",
        "make_zero",
        "-y",
        str(dest),
    ]


def require_non_empty(text, field_name):
    value = str(text or "").strip()
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")
    return value


def apply_config_overrides(config, **overrides):
    cleaned = {k: v for k, v in overrides.items() if v is not None}
    if not cleaned:
        return config
    return dataclass_replace(config, **cleaned)


from fractions import Fraction


def _np():
    import numpy as np  # type: ignore

    return np


def _parse_timebase_fraction(text):
    raw = str(text or "").strip()
    if "/" in raw:
        num_s, den_s = raw.split("/", 1)
        num = int(num_s)
        den = int(den_s)
    else:
        num = int(raw)
        den = 1
    if den == 0:
        raise ValueError("timebase denominator cannot be zero")
    if den < 0:
        num = -num
        den = -den
    return int(num), int(den)


def _round_fraction_nearest_int(frac):
    frac = Fraction(frac)
    if frac >= 0:
        return int((frac.numerator * 2 + frac.denominator) // (2 * frac.denominator))
    pos = -frac
    return -int((pos.numerator * 2 + pos.denominator) // (2 * pos.denominator))


def chapter_frame_bounds(chapter, fps_num=30000, fps_den=1001):
    """
    Return chapter [start_frame, end_frame) using exact rational math when raw
    ffmetadata ticks/timebase are available. Falls back to float seconds.
    """
    fps = Fraction(int(fps_num), int(fps_den))
    try:
        s_raw = int(chapter.get("start_raw"))
        e_raw = int(chapter.get("end_raw"))
        tb_num = int(chapter.get("timebase_num"))
        tb_den = int(chapter.get("timebase_den"))
        tb = Fraction(tb_num, tb_den)
        s = _round_fraction_nearest_int(Fraction(s_raw) * tb * fps)
        e = _round_fraction_nearest_int(Fraction(e_raw) * tb * fps)
    except Exception:
        s = int(round(float(chapter.get("start", 0.0)) * float(fps_num) / float(fps_den)))
        e = int(round(float(chapter.get("end", 0.0)) * float(fps_num) / float(fps_den)))
    if e < s:
        e = s
    return int(s), int(e)


def parse_chapters(path):
    chapters = []
    ffmetadata = {}
    cur = {}
    in_chapter = False
    seen_chapter = False

    def finalize(ch):
        tb_num = 1
        tb_den = 1
        if "timebase" in ch:
            tb_num, tb_den = _parse_timebase_fraction(ch["timebase"])
        tb = Fraction(int(tb_num), int(tb_den))
        ch["timebase_num"] = int(tb_num)
        ch["timebase_den"] = int(tb_den)

        if "start" in ch:
            s_raw = int(ch["start"])
            ch["start_raw"] = int(s_raw)
            ch["start"] = float(Fraction(s_raw) * tb)
        else:
            ch["start"] = float(ch.get("start", 0.0))

        if "end" in ch:
            e_raw = int(ch["end"])
            ch["end_raw"] = int(e_raw)
            ch["end"] = float(Fraction(e_raw) * tb)
        else:
            ch["end"] = float(ch.get("end", 0.0))

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # global ffmetadata before any chapter
        if not seen_chapter and "=" in line and not line.startswith(("[", ";")):
            k, v = line.split("=", 1)
            ffmetadata[k.strip().lower()] = v.strip()
            continue

        if line == "[CHAPTER]":
            # finish previous chapter
            if cur and in_chapter:
                finalize(cur)
                chapters.append(cur)
            cur = {}
            in_chapter = True
            seen_chapter = True
            continue

        if in_chapter and "=" in line:
            k, v = line.split("=", 1)
            cur[k.lower()] = v.strip()

    # finalize last chapter
    if cur and in_chapter:
        finalize(cur)
        chapters.append(cur)

    return ffmetadata, chapters


def parse_bad_frames_csv(text):
    vals = []
    seen = set()
    for raw in str(text or "").split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            fid = int(token)
        except Exception:
            continue
        if fid < 0 or fid in seen:
            continue
        seen.add(fid)
        vals.append(fid)
    vals.sort()
    return vals


def robust_zscore(values, return_stats=False):
    np = _np()
    vals = np.asarray(values, dtype=np.float64)
    center = float(np.median(vals))
    mad = float(np.median(np.abs(vals - center)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        std = float(np.std(vals))
        scale = std if std > 1e-12 else 1.0
    z = (vals - center) / scale
    if bool(return_stats):
        return z, center, scale
    return z


def combine_signal_scores(
    chroma_scores,
    noise_scores,
    tear_scores,
    wave_scores,
    weight_chroma,
    weight_noise,
    weight_tear,
    weight_wave,
    include_norm=False,
):
    np = _np()
    w_sum = float(weight_chroma) + float(weight_noise) + float(weight_tear) + float(weight_wave)
    if w_sum <= 0:
        raise ValueError("At least one signal weight must be > 0.")
    chroma_z, cc, cs = robust_zscore(chroma_scores, return_stats=True)
    noise_z, nc, ns = robust_zscore(noise_scores, return_stats=True)
    tear_z, tc, ts = robust_zscore(tear_scores, return_stats=True)
    wave_z, wc, ws = robust_zscore(wave_scores, return_stats=True)
    score = (
        float(weight_chroma) * chroma_z
        + float(weight_noise) * noise_z
        + float(weight_tear) * tear_z
        + float(weight_wave) * wave_z
    ) / w_sum
    score = score.astype(np.float64)
    if not bool(include_norm):
        return score
    norm = {
        "chroma": {"center": float(cc), "scale": float(cs)},
        "noise": {"center": float(nc), "scale": float(ns)},
        "tear": {"center": float(tc), "scale": float(ts)},
        "wave": {"center": float(wc), "scale": float(ws)},
    }
    return score, norm


def combined_score(sigs, wc, wn, wt, ww):
    return combine_signal_scores(
        sigs["chroma"],
        sigs["noise"],
        sigs["tear"],
        sigs["wave"],
        wc,
        wn,
        wt,
        ww,
        include_norm=False,
    )


def compute_threshold(scores, mode, iqr_mult, thresh_val, bad_pct):
    np = _np()
    v = np.asarray(scores, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    mode_n = str(mode or "").strip().lower()
    if mode_n == "iqr":
        q1, q3 = float(np.percentile(v, 25)), float(np.percentile(v, 75))
        return q3 + float(iqr_mult) * (q3 - q1)
    if mode_n == "value":
        return float(thresh_val)
    return float(np.quantile(v, 1.0 - float(bad_pct) / 100.0))


def format_bad_frames_csv(frame_ids):
    vals = sorted({int(x) for x in (frame_ids or []) if int(x) >= 0})
    return ",".join(str(v) for v in vals)


def render_settings_path(archive: str) -> Path:
    return METADATA_DIR / str(archive or "").strip() / "render_settings.json"


GAMMA_CORRECTION_DEFAULT_KEY = "gamma_correction_default"
GAMMA_CORRECTION_RANGES_KEY = "gamma_correction_ranges"
AUDIO_SYNC_OFFSETS_KEY = "audio_sync_offsets"


def _render_settings_template() -> dict:
    return {
        "version": 3,
        "_comments": {
            "archive_settings": ("Archive-wide defaults applied to render behavior for all chapters."),
            "bad_frames": ("Global archive frame IDs that are bad and will be repaired."),
            "gamma_correction_ranges": (
                "Gamma correction ranges use global frame IDs: start_frame inclusive, end_frame exclusive."
            ),
            "audio_sync_offsets": (
                "Per-range audio sync offsets in seconds (positive = delay audio, negative = advance audio). "
                "Ranges use global archive frame IDs: start_frame inclusive, end_frame exclusive."
            ),
        },
        "archive_settings": {
            "gamma_correction_default": 1.0,
            "gamma_correction_ranges": [],
        },
        "bad_frames": [],
        "audio_sync_offsets": [],
    }


def _normalize_transcript_mode(raw: object, default: str = "off") -> str:
    mode = str(raw if raw is not None else default).strip().lower()
    if mode in {"off", "false", "0", "no", "skip", "none"}:
        return "off"
    if mode in {"on", "true", "1", "yes", "force", "auto"}:
        return "on"
    return str(default).strip().lower() if str(default).strip() else "off"


def normalize_gamma_value(raw: object, default: float = 1.0) -> float:
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    if not (value == value):  # NaN
        value = float(default)
    return max(0.05, min(8.0, float(value)))


def _canonicalize_gamma_ranges(raw_ranges) -> list[dict[str, float | int]]:
    entries: list[tuple[int, int, float, int]] = []
    for idx, item in enumerate(list(raw_ranges or [])):
        start = end = None
        gamma = None
        if isinstance(item, dict):
            start = item.get("start_frame")
            end = item.get("end_frame")
            gamma = item.get("gamma")
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start = item[0]
            end = item[1]
            gamma = item[2]
        try:
            a = int(start)
            b = int(end)
        except Exception:
            continue
        if b <= a:
            continue
        g = normalize_gamma_value(gamma, default=1.0)
        entries.append((a, b, g, idx))
    if not entries:
        return []

    boundaries = set()
    for a, b, _g, _idx in entries:
        boundaries.add(int(a))
        boundaries.add(int(b))
    cuts = sorted(boundaries)
    if len(cuts) < 2:
        return []

    resolved: list[tuple[int, int, float]] = []
    for i in range(len(cuts) - 1):
        seg_a = int(cuts[i])
        seg_b = int(cuts[i + 1])
        if seg_b <= seg_a:
            continue
        winner_idx = -1
        winner_gamma = None
        for a, b, g, idx in entries:
            if a <= seg_a and seg_b <= b and idx >= winner_idx:
                winner_idx = idx
                winner_gamma = g
        if winner_gamma is None:
            continue
        if resolved and resolved[-1][1] == seg_a and abs(float(resolved[-1][2]) - float(winner_gamma)) < 1e-6:
            prev_a, _prev_b, prev_g = resolved[-1]
            resolved[-1] = (prev_a, seg_b, prev_g)
        else:
            resolved.append((seg_a, seg_b, float(winner_gamma)))

    return [
        {
            "start_frame": int(a),
            "end_frame": int(b),
            "gamma": round(float(g), 4),
        }
        for a, b, g in resolved
        if int(b) > int(a)
    ]


def _clip_gamma_ranges_to_span(
    ranges,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> list[dict[str, float | int]]:
    if start_frame is None or end_frame is None:
        return _canonicalize_gamma_ranges(ranges)

    start = int(start_frame)
    end = int(end_frame)
    if end <= start:
        return []
    clipped = []
    for item in _canonicalize_gamma_ranges(ranges):
        a = int(item["start_frame"])
        b = int(item["end_frame"])
        g = float(item["gamma"])
        ra = max(start, a)
        rb = min(end, b)
        if rb <= ra:
            continue
        clipped.append({"start_frame": int(ra), "end_frame": int(rb), "gamma": float(g)})
    return _canonicalize_gamma_ranges(clipped)


def _gamma_default_from_cfg(cfg: dict, default: float = 1.0) -> float:
    if not isinstance(cfg, dict):
        return normalize_gamma_value(default, default=1.0)
    if GAMMA_CORRECTION_DEFAULT_KEY in cfg:
        return normalize_gamma_value(cfg.get(GAMMA_CORRECTION_DEFAULT_KEY), default=default)
    return normalize_gamma_value(default, default=1.0)


def _gamma_ranges_from_cfg(cfg: dict) -> list[dict[str, float | int]]:
    if not isinstance(cfg, dict):
        return []
    if GAMMA_CORRECTION_RANGES_KEY in cfg:
        return _canonicalize_gamma_ranges(cfg.get(GAMMA_CORRECTION_RANGES_KEY, []))
    return []


def _migrate_v1_to_v2(data: dict) -> dict:
    """Migrate a v1 render_settings dict to v2 in memory."""
    out = dict(_render_settings_template())
    old_archive = dict(data.get("archive_settings") or {})
    old_chapter_settings = dict(data.get("chapter_settings") or {})
    old_bad_by_chapter = dict(data.get("bad_frames_by_chapter") or {})

    # Archive gamma: carry forward
    out["archive_settings"][GAMMA_CORRECTION_DEFAULT_KEY] = _gamma_default_from_cfg(old_archive, default=1.0)
    archive_ranges = list(_gamma_ranges_from_cfg(old_archive))

    # Merge chapter-level gamma overrides into archive ranges
    for _, raw_cfg in old_chapter_settings.items():
        cfg = dict(raw_cfg or {}) if isinstance(raw_cfg, dict) else {}
        ch_ranges = _gamma_ranges_from_cfg(cfg)
        if ch_ranges:
            archive_ranges = archive_ranges + ch_ranges

    out["archive_settings"][GAMMA_CORRECTION_RANGES_KEY] = _canonicalize_gamma_ranges(archive_ranges)

    # Flatten bad_frames_by_chapter → sorted deduplicated bad_frames list
    merged: set[int] = set()
    for vals in old_bad_by_chapter.values():
        for item in list(vals or []):
            try:
                fid = int(item)
            except Exception:
                continue
            if fid >= 0:
                merged.add(fid)
    out["bad_frames"] = sorted(merged)

    return out


def _canonicalize_audio_sync_offsets(raw_offsets) -> list[dict]:
    """Normalize audio_sync_offsets to sorted, non-overlapping frame ranges."""
    entries = []
    for item in list(raw_offsets or []):
        if not isinstance(item, dict):
            continue
        try:
            a = int(item["start_frame"])
            b = int(item["end_frame"])
            offset = float(item["offset_seconds"])
        except Exception:
            continue
        if b <= a:
            continue
        entries.append((a, b, offset))
    entries.sort(key=lambda e: e[0])
    # Merge adjacent entries with identical offset
    merged = []
    for a, b, offset in entries:
        if merged and merged[-1][1] == a and abs(merged[-1][2] - offset) < 1e-9:
            prev_a, _prev_b, prev_off = merged[-1]
            merged[-1] = (prev_a, b, prev_off)
        else:
            merged.append((a, b, offset))
    return [{"start_frame": int(a), "end_frame": int(b), "offset_seconds": round(offset, 4)} for a, b, offset in merged]


def _migrate_v2_to_v3(data: dict) -> dict:
    """Migrate a v2 render_settings dict to v3 in memory (adds audio_sync_offsets)."""
    out = dict(data)
    out["version"] = 3
    if "audio_sync_offsets" not in out:
        out["audio_sync_offsets"] = []
    return out


def load_render_settings(archive: str, create: bool = False) -> tuple[Path, dict]:
    path = render_settings_path(archive)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                version = int(data.get("version") or 1)
                if version < 2:
                    out = _migrate_v1_to_v2(data)
                else:
                    out = dict(_render_settings_template())
                    out.update(data)
                if int(out.get("version") or 2) < 3:
                    out = _migrate_v2_to_v3(out)
                out["archive_settings"] = dict(out.get("archive_settings") or {})
                out["archive_settings"][GAMMA_CORRECTION_DEFAULT_KEY] = _gamma_default_from_cfg(
                    out["archive_settings"],
                    default=1.0,
                )
                out["archive_settings"][GAMMA_CORRECTION_RANGES_KEY] = _gamma_ranges_from_cfg(
                    out["archive_settings"],
                )
                bad = out.get("bad_frames") or []
                out["bad_frames"] = sorted({int(x) for x in bad if int(x) >= 0})
                out[AUDIO_SYNC_OFFSETS_KEY] = _canonicalize_audio_sync_offsets(out.get(AUDIO_SYNC_OFFSETS_KEY) or [])
                return path, out
        except Exception:
            pass
    data = _render_settings_template()
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path, data


def save_render_settings(archive: str, settings: dict) -> Path:
    path = render_settings_path(archive)
    payload = dict(_render_settings_template())
    payload.update(dict(settings or {}))
    payload["_comments"] = dict(_render_settings_template().get("_comments") or {})
    payload["archive_settings"] = dict(payload.get("archive_settings") or {})
    payload["archive_settings"][GAMMA_CORRECTION_DEFAULT_KEY] = _gamma_default_from_cfg(
        payload["archive_settings"],
        default=1.0,
    )
    payload["archive_settings"][GAMMA_CORRECTION_RANGES_KEY] = _gamma_ranges_from_cfg(
        payload["archive_settings"],
    )
    bad = payload.get("bad_frames") or []
    payload["bad_frames"] = sorted({int(x) for x in bad if int(x) >= 0})
    payload[AUDIO_SYNC_OFFSETS_KEY] = _canonicalize_audio_sync_offsets(payload.get(AUDIO_SYNC_OFFSETS_KEY) or [])
    # Drop any stale v1 keys
    payload.pop("chapter_settings", None)
    payload.pop("bad_frames_by_chapter", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_bad_frames_from_render_settings(archive: str) -> list[int]:
    _path, settings = load_render_settings(archive, create=False)
    return list(settings.get("bad_frames") or [])


def get_audio_sync_offset_for_chapter(
    archive: str,
    ch_start: int,
    ch_end: int,
) -> float:
    """Return the audio sync offset (seconds) for the frame range [ch_start, ch_end).

    Uses the last matching entry whose range overlaps the chapter midpoint.
    Returns 0.0 if no entry applies.
    """
    _path, settings = load_render_settings(archive, create=False)
    offsets = _canonicalize_audio_sync_offsets(settings.get(AUDIO_SYNC_OFFSETS_KEY) or [])
    if not offsets:
        return 0.0
    mid = (int(ch_start) + int(ch_end)) // 2
    result = 0.0
    for entry in offsets:
        a = int(entry["start_frame"])
        b = int(entry["end_frame"])
        if a <= mid < b:
            result = float(entry["offset_seconds"])
    return result


def update_chapter_audio_sync_in_render_settings(
    archive: str,
    ch_start: int,
    ch_end: int,
    offset_seconds: float,
) -> Path:
    """Set (or clear) the audio sync offset for the frame range [ch_start, ch_end)."""
    _, settings = load_render_settings(archive, create=True)
    existing = _canonicalize_audio_sync_offsets(settings.get(AUDIO_SYNC_OFFSETS_KEY) or [])
    # Remove any existing entries that overlap [ch_start, ch_end)
    kept = []
    for entry in existing:
        a = int(entry["start_frame"])
        b = int(entry["end_frame"])
        if b <= ch_start or a >= ch_end:
            kept.append(entry)
        else:
            # Preserve portions outside the chapter span
            if a < ch_start:
                kept.append(
                    {
                        "start_frame": a,
                        "end_frame": ch_start,
                        "offset_seconds": entry["offset_seconds"],
                    }
                )
            if b > ch_end:
                kept.append(
                    {
                        "start_frame": ch_end,
                        "end_frame": b,
                        "offset_seconds": entry["offset_seconds"],
                    }
                )
    # Add the new entry only if offset is non-zero
    if abs(float(offset_seconds)) >= 1e-6:
        kept.append(
            {
                "start_frame": int(ch_start),
                "end_frame": int(ch_end),
                "offset_seconds": float(offset_seconds),
            }
        )
    settings[AUDIO_SYNC_OFFSETS_KEY] = _canonicalize_audio_sync_offsets(kept)
    return save_render_settings(archive, settings)


def get_bad_frames_for_chapter(
    archive: str,
    chapter_title: str,
    *,
    ch_start: int | None = None,
    ch_end: int | None = None,
) -> list[int]:
    all_bad = load_bad_frames_from_render_settings(archive)
    if not all_bad:
        return []
    if ch_start is not None and ch_end is not None:
        return [f for f in all_bad if ch_start <= f < ch_end]
    title = str(chapter_title or "").strip()
    if not title:
        return []
    for ext in ("chapters.tsv", "chapters.ffmetadata"):
        chapters_path = METADATA_DIR / str(archive or "").strip() / ext
        if chapters_path.exists():
            _, chapters = parse_chapters(chapters_path)
            chapter_obj = next(
                (ch for ch in chapters if str(ch.get("title", "")).strip() == title),
                None,
            )
            if chapter_obj is not None:
                start_frame, end_frame = chapter_frame_bounds(chapter_obj, fps_num=30000, fps_den=1001)
                return [f for f in all_bad if start_frame <= f < end_frame]
    return []


def merge_bad_frames_in_render_settings(archive: str, new_frames) -> Path:
    _path, settings = load_render_settings(archive, create=True)
    existing = set(settings.get("bad_frames") or [])
    for item in list(new_frames or []):
        try:
            fid = int(item)
        except Exception:
            continue
        if fid >= 0:
            existing.add(fid)
    settings["bad_frames"] = sorted(existing)
    return save_render_settings(archive, settings)


def replace_chapter_bad_frames_in_render_settings(
    archive: str,
    ch_start: int,
    ch_end: int,
    new_frames: list[int],
) -> Path:
    _, settings = load_render_settings(archive, create=True)
    existing = [f for f in (settings.get("bad_frames") or []) if not (ch_start <= f < ch_end)]
    new_valid = sorted({int(f) for f in (new_frames or []) if ch_start <= int(f) < ch_end})
    settings["bad_frames"] = sorted(set(existing) | set(new_valid))
    return save_render_settings(archive, settings)


def get_gamma_profile_for_chapter(
    archive: str,
    _chapter_title: str = "",
    *,
    ch_start: int | None = None,
    ch_end: int | None = None,
) -> dict[str, object]:
    _path, settings = load_render_settings(archive, create=False)
    archive_settings = dict(settings.get("archive_settings") or {})

    effective_default = _gamma_default_from_cfg(archive_settings, default=1.0)
    effective_ranges = _gamma_ranges_from_cfg(archive_settings)

    if ch_start is not None and ch_end is not None:
        effective_ranges = _clip_gamma_ranges_to_span(
            effective_ranges,
            start_frame=int(ch_start),
            end_frame=int(ch_end),
        )
    else:
        effective_ranges = _canonicalize_gamma_ranges(effective_ranges)

    source = "archive" if effective_ranges else "default"

    return {
        "default_gamma": float(effective_default),
        "ranges": effective_ranges,
        "source": source,
    }


def _clip_ranges_outside_span(
    ranges: list[dict],
    ch_start: int,
    ch_end: int,
) -> list[dict]:
    """Return archive ranges with the [ch_start, ch_end) span excised."""
    result = []
    for r in ranges:
        a, b, g = int(r["start_frame"]), int(r["end_frame"]), float(r["gamma"])
        if a < ch_start:
            result.append({"start_frame": a, "end_frame": min(b, ch_start), "gamma": g})
        if b > ch_end:
            result.append({"start_frame": max(a, ch_end), "end_frame": b, "gamma": g})
    return result


def update_chapter_gamma_in_render_settings(
    archive: str,
    _chapter_title: str = "",
    *,
    ch_start: int | None = None,
    ch_end: int | None = None,
    gamma_ranges,
    default_gamma: float | None = None,
) -> Path:
    _, settings = load_render_settings(archive, create=True)
    archive_settings = dict(settings.get("archive_settings") or {})
    archive_default = _gamma_default_from_cfg(archive_settings, default=1.0)
    existing_ranges = _gamma_ranges_from_cfg(archive_settings)

    # Remove archive ranges within the chapter span so the new ones take over
    if ch_start is not None and ch_end is not None:
        outside = _clip_ranges_outside_span(existing_ranges, ch_start, ch_end)
    else:
        outside = list(existing_ranges)

    new_ranges = list(_canonicalize_gamma_ranges(gamma_ranges))

    # If chapter default differs from archive default, add a base range for the span
    if default_gamma is not None and ch_start is not None and ch_end is not None:
        next_default = normalize_gamma_value(default_gamma, default=archive_default)
        if abs(next_default - float(archive_default)) >= 1e-6:
            new_ranges = [{"start_frame": ch_start, "end_frame": ch_end, "gamma": next_default}] + new_ranges

    merged = _canonicalize_gamma_ranges(outside + new_ranges)
    archive_settings[GAMMA_CORRECTION_RANGES_KEY] = merged
    archive_settings[GAMMA_CORRECTION_DEFAULT_KEY] = float(archive_default)
    settings["archive_settings"] = archive_settings
    return save_render_settings(archive, settings)


def get_transcript_mode_for_chapter(archive: str, chapter_title: str) -> str:
    import csv as _csv

    tsv_path = METADATA_DIR / str(archive or "").strip() / "chapters.tsv"
    if not tsv_path.exists():
        return "off"
    title = str(chapter_title or "").strip()
    with tsv_path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
        for row in _csv.DictReader(fh, delimiter="\t"):
            if str((row or {}).get("title", "")).strip() == title:
                return _normalize_transcript_mode((row or {}).get("transcript", "off"))
    return "off"


def update_chapter_transcript_in_chapters_tsv(
    archive: str,
    chapter_title: str,
    *,
    transcript: str,
) -> Path:
    import csv as _csv

    tsv_path = METADATA_DIR / str(archive or "").strip() / "chapters.tsv"
    if not tsv_path.exists():
        return tsv_path
    title = str(chapter_title or "").strip()
    normalized = _normalize_transcript_mode(transcript)
    with tsv_path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
        reader = _csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "transcript" not in fieldnames:
        fieldnames.append("transcript")
    updated = False
    for row in rows:
        if str((row or {}).get("title", "")).strip() == title:
            row["transcript"] = normalized
            updated = True
    if not updated:
        return tsv_path
    with tsv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = _csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return tsv_path


_FRAME_LIST_KEYS = (
    "bad_frames",
    "bad_frame_override",
    "good_frame_override",
)


def _normalize_frame_list_key(key):
    k = str(key or "").strip().lower()
    return k if k in _FRAME_LIST_KEYS else ""


def update_chapter_frame_lists_in_ffmetadata(path, chapter_frame_lists):
    """
    Update chapter frame-list metadata lines in chapters.ffmetadata in-place.
    chapter_frame_lists:
      {
        chapter_title: {
          "BAD_FRAMES": [...],
          "BAD_FRAME_OVERRIDE": [...],
          "GOOD_FRAME_OVERRIDE": [...],
        }
      }
    Empty lists remove the corresponding key from that chapter block.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"chapters.ffmetadata not found: {p}")

    def _norm_title(text):
        return " ".join(str(text or "").strip().lower().split())

    pending = {}
    for raw_title, raw_fields in (chapter_frame_lists or {}).items():
        nk = _norm_title(raw_title)
        if not nk:
            continue
        field_map = {}
        for raw_key, vals in dict(raw_fields or {}).items():
            key = _normalize_frame_list_key(raw_key)
            if not key:
                continue
            field_map[key] = list(vals or [])
        if field_map:
            pending[nk] = field_map
    if not pending:
        return 0

    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    i = 0
    touched = 0

    while i < len(lines):
        line = lines[i]
        if line.strip() != "[CHAPTER]":
            out.append(line)
            i += 1
            continue

        block = [line]
        i += 1
        while i < len(lines) and lines[i].strip() != "[CHAPTER]":
            block.append(lines[i])
            i += 1

        title = ""
        for bline in block:
            s = bline.strip()
            if "=" in s and not s.startswith(";"):
                k, v = s.split("=", 1)
                if k.strip().lower() == "title":
                    title = v.strip()
                    break

        nk = _norm_title(title)
        updates = pending.get(nk)
        should_update = updates is not None
        remove_keys = set(updates.keys()) if should_update else set()

        title_idx = -1
        cleaned = []
        for bline in block:
            s = bline.strip()
            if "=" in s and not s.startswith(";"):
                k, _v = s.split("=", 1)
                key = k.strip().lower()
                if key == "title":
                    title_idx = len(cleaned)
                if should_update and key in remove_keys:
                    continue
            cleaned.append(bline)

        if should_update:
            insert_at = title_idx + 1 if title_idx >= 0 else len(cleaned)
            for key in _FRAME_LIST_KEYS:
                if key not in updates:
                    continue
                csv = format_bad_frames_csv(updates[key])
                if csv:
                    cleaned.insert(insert_at, f"{key.upper()}={csv}")
                    insert_at += 1
            touched += 1

        out.extend(cleaned)

    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return touched


def update_chapter_bad_frames_in_ffmetadata(path, chapter_bad_frames):
    """
    Update BAD_FRAMES lines in chapters.ffmetadata in-place.
    chapter_bad_frames: {chapter_title: [global_frame_ids]}.
    """
    mapped = {title: {"BAD_FRAMES": list(vals or [])} for title, vals in (chapter_bad_frames or {}).items()}
    return update_chapter_frame_lists_in_ffmetadata(path, mapped)


def is_chapter_done(final_file):
    if not final_file.exists():
        return False

    if final_file.stat().st_size < 100_000:
        return False

    return True


def duration(path):
    try:
        out = subprocess.check_output(
            [
                FFPROBE_BIN,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
        ).strip()
        return float(out) if out else 0
    except:
        return 0


# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------


def ensure_ffmpeg_exists():
    if not _command_exists(FFMPEG_BIN):
        raise FileNotFoundError(f"FFmpeg not found at {FFMPEG_BIN}")


# ---------------------------------------------------------
# SHA3-256 Checksums
# ---------------------------------------------------------


def sha3sum_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha3_256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_sha3_manifest(root_dir, manifest_path, relative_base=None, ignore_fn=None):
    root_dir = Path(root_dir)
    manifest_path = Path(manifest_path)
    relative_base = Path(relative_base) if relative_base else root_dir

    manifest_path.unlink(missing_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "a", encoding="utf-8") as out:
        for file_path in root_dir.rglob("*"):
            if ignore_fn and ignore_fn(file_path):
                continue

            if not file_path.is_file():
                continue

            if file_path.resolve() == manifest_path.resolve():
                continue

            digest = sha3sum_file(file_path)
            rel_path = file_path.relative_to(relative_base)
            out.write(f"{digest}  {rel_path}\n")

    print("Checksums written to:", manifest_path)


def verify_sha3_manifest(root_dir, manifest_path):
    root_dir = Path(root_dir)
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    failures = 0
    total = 0

    with manifest_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"Skipping malformed line: {raw_line.rstrip()}")
                failures += 1
                continue

            expected, rel_path = parts
            rel_path = rel_path.lstrip("*")
            target = root_dir / rel_path

            if not target.exists():
                print(f"MISSING: {rel_path}")
                failures += 1
                continue

            actual = sha3sum_file(target)
            total += 1

            if actual.lower() != expected.lower():
                print(f"MISMATCH: {rel_path}")
                failures += 1

    if failures == 0:
        print("ALL FILES VERIFIED - CHECKSUMS MATCH!")
        return 0

    print(f"{failures} FILES FAILED VERIFICATION")
    return 1


def verify_blake3_manifest(root_dir, manifest_path):
    if not _command_exists(B3SUM_BIN):
        print(f"ERROR: b3sum not found at {B3SUM_BIN}")
        return 1

    r = subprocess.run(
        [str(B3SUM_BIN), "-c", str(manifest_path)],
        cwd=Path(root_dir),
        capture_output=True,
        text=True,
    )
    print(r.stdout or r.stderr)

    if r.returncode == 0:
        print("ALL FILES VERIFIED - CHECKSUMS MATCH!")
    else:
        print("SOME FILES FAILED VERIFICATION!")

    return r.returncode


def detect_manifest_algo(manifest_path):
    name = Path(manifest_path).name.lower()
    if "blake3" in name:
        return "blake3"
    if "sha3" in name:
        return "sha3"
    return None


def verify_manifest(root_dir, manifest_path, algo="auto"):
    algo = (algo or "auto").lower()
    manifest_path = Path(manifest_path)

    if algo == "auto":
        algo = detect_manifest_algo(manifest_path) or "sha3"

    if algo == "blake3":
        return verify_blake3_manifest(root_dir, manifest_path)

    if algo == "sha3":
        return verify_sha3_manifest(root_dir, manifest_path)

    print(f"Unknown checksum algorithm: {algo}")
    return 1
