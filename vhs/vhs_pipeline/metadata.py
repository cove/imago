from __future__ import annotations

import csv
import glob
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import Any

import importlib.metadata

from common import (
    FFMPEG_BIN,
    MEDIAINFO_BIN,
    METADATA_DIR,
    all_store_archive_dirs,
    archive_checksum_file_for,
    archive_dir_for,
    write_sha3_manifest,
)
from vhs_pipeline.render_pipeline import (
    ENCODE_AUDIO_BITRATE,
    ENCODE_AUDIO_CHANNELS,
    ENCODE_AUDIO_CODEC,
    ENCODE_AUDIO_FILTERS,
    ENCODE_AUDIO_SAMPLE_RATE,
    ENCODE_VIDEO_CODEC,
    ENCODE_VIDEO_CRF,
    ENCODE_VIDEO_LEVEL,
    ENCODE_VIDEO_PIX_FMT,
    ENCODE_VIDEO_PRESET,
    ENCODE_VIDEO_PROFILE,
    ENCODE_VIDEO_TUNE,
    WHISPER_AUDIO_FILTERS,
    WHISPER_MODEL,
)

TSV_META_CHAPTER_INDEX_COL = "__chapter_index"
TSV_FFMETA_PREFIX = "ffmeta_"
TSV_META_PREFIX = "__"


def _as_text(value: Any) -> str:
    return str(value if value is not None else "")


def _safe_int(value: Any) -> int | None:
    text = _as_text(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return None


def _parse_timebase(value: str) -> tuple[int, int] | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if "/" in raw:
        num_s, den_s = raw.split("/", 1)
    else:
        num_s, den_s = raw, "1"
    try:
        num = int(num_s.strip())
        den = int(den_s.strip())
    except Exception:
        return None
    if den == 0:
        return None
    if den < 0:
        num = -num
        den = -den
    return int(num), int(den)


def _sort_rows_by_index(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    indexed: list[tuple[int, int, dict[str, str]]] = []
    for pos, row in enumerate(list(rows or [])):
        idx = _safe_int((row or {}).get(TSV_META_CHAPTER_INDEX_COL))
        if idx is None:
            idx = pos + 1
        indexed.append((int(idx), int(pos), dict(row or {})))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in indexed]


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
            row = {col: _as_text((raw_row or {}).get(col, "")) for col in header}
            if any(_as_text(v).strip() for v in row.values()):
                rows.append(row)
    return header, rows


def _write_chapters_tsv_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
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
            writer.writerow({col: _as_text((raw_row or {}).get(col, "")) for col in header})


def _parse_ffmetadata_raw(
    path: Path,
) -> tuple[str, list[tuple[str, str]], list[list[tuple[str, str]]]]:
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    header = ";FFMETADATA1"
    globals_list: list[tuple[str, str]] = []
    chapters: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] | None = None

    for raw in lines:
        line = str(raw or "").rstrip("\r\n")
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith(";") and current is None and not globals_list:
            header = stripped
            continue
        if stripped == "[CHAPTER]":
            if current is not None:
                chapters.append(current)
            current = []
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if current is None:
            globals_list.append((key, value))
        else:
            current.append((key, value))
    if current is not None:
        chapters.append(current)
    return header, globals_list, chapters


def ffmetadata_to_chapters_tsv(ffmetadata_path: Path, out_path: Path | None = None) -> Path:
    source = Path(ffmetadata_path)
    target = Path(out_path) if out_path is not None else (source.parent / "chapters.tsv")
    header_line, globals_list, chapter_lists = _parse_ffmetadata_raw(source)

    global_order: list[str] = []
    global_values: dict[str, str] = {}
    for key, value in globals_list:
        if key not in global_order:
            global_order.append(key)
        global_values[key] = value

    chapter_columns: list[str] = []
    for chapter in list(chapter_lists or []):
        seen_keys: set[str] = set()
        for key, _value in list(chapter or []):
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if key not in chapter_columns:
                chapter_columns.append(key)

    rows: list[dict[str, Any]] = []
    chapter_rows = list(chapter_lists or [])
    if not chapter_rows:
        chapter_rows = [[]]

    for idx, chapter in enumerate(chapter_rows, start=1):
        chapter_values: dict[str, str] = {}
        for key, value in list(chapter or []):
            chapter_values[key] = value

        row: dict[str, Any] = {TSV_META_CHAPTER_INDEX_COL: str(int(idx))}
        for key in global_order:
            row[f"{TSV_FFMETA_PREFIX}{key}"] = global_values.get(key, "")
        for key in chapter_columns:
            row[key] = chapter_values.get(key, "")
        rows.append(row)

    columns = [
        TSV_META_CHAPTER_INDEX_COL,
        *[f"{TSV_FFMETA_PREFIX}{key}" for key in global_order],
        *chapter_columns,
    ]
    _write_chapters_tsv_rows(target, columns, rows)
    print(f"  Generated chapters master TSV: {target}")
    return target


def convert_all_ffmetadata_to_chapters_tsv(
    metadata_root: Path = METADATA_DIR,
    *,
    overwrite: bool = False,
) -> int:
    root = Path(metadata_root)
    if not root.exists():
        print(f"Metadata root not found: {root}")
        return 0
    count = 0
    for archive_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        ffmeta = archive_dir / "chapters.ffmetadata"
        tsv = archive_dir / "chapters.tsv"
        if not ffmeta.exists():
            continue
        if tsv.exists() and not overwrite:
            continue
        ffmetadata_to_chapters_tsv(ffmeta, tsv)
        count += 1
    print(f"Generated chapters.tsv files: {count}")
    return int(count)


def _chapter_keys_for_row(header: list[str]) -> list[str]:
    return [
        key
        for key in list(header or [])
        if key and not key.startswith(TSV_META_PREFIX) and not key.startswith(TSV_FFMETA_PREFIX)
    ]


def generate_ffmetadata_from_chapters_tsv(chapters_tsv_path: Path, out_path: Path) -> Path:
    header, rows = _read_chapters_tsv_rows(Path(chapters_tsv_path))
    rows_sorted = _sort_rows_by_index(rows)

    if not rows_sorted:
        Path(out_path).write_text(";FFMETADATA1\n", encoding="utf-8")
        print(f"  Generated FFmetadata: {out_path}")
        return Path(out_path)

    global_order: list[str] = [
        str(col)[len(TSV_FFMETA_PREFIX) :] for col in list(header or []) if str(col or "").startswith(TSV_FFMETA_PREFIX)
    ]

    global_values: dict[str, str] = {}
    for key in global_order:
        col = f"{TSV_FFMETA_PREFIX}{key}"
        chosen = ""
        for row in rows_sorted:
            value = _as_text((row or {}).get(col, ""))
            chosen = value
            if value != "":
                break
        global_values[key] = chosen

    lines: list[str] = [";FFMETADATA1"]
    for key in global_order:
        lines.append(f"{key}={global_values.get(key, '')}")
    if rows_sorted:
        lines.append("")

    for idx, row in enumerate(rows_sorted):
        lines.append("[CHAPTER]")
        for key in list(header or []):
            key_text = str(key or "").strip()
            if not key_text or key_text.startswith(TSV_META_PREFIX) or key_text.startswith(TSV_FFMETA_PREFIX):
                continue
            value = _as_text((row or {}).get(key_text, ""))
            if value != "":
                lines.append(f"{key_text}={value}")

        if idx < len(rows_sorted) - 1:
            lines.append("")

    out = Path(out_path)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Generated FFmetadata: {out}")
    return out


def _load_master_chapters(
    chapters_tsv_path: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    header, rows = _read_chapters_tsv_rows(chapters_tsv_path)
    rows_sorted = _sort_rows_by_index(rows)
    ffmeta: dict[str, str] = {}
    for col in list(header or []):
        key = str(col or "")
        if not key.startswith(TSV_FFMETA_PREFIX):
            continue
        name = key[len(TSV_FFMETA_PREFIX) :].strip().lower()
        if not name:
            continue
        chosen = ""
        for row in rows_sorted:
            value = _as_text((row or {}).get(key, ""))
            chosen = value
            if value != "":
                break
        ffmeta[name] = chosen

    chapters: list[dict[str, Any]] = []
    for row in rows_sorted:
        chapter_keys = _chapter_keys_for_row(header)
        chapter: dict[str, Any] = {}
        for key in chapter_keys:
            key_text = str(key or "").strip()
            if not key_text or key_text.startswith(TSV_META_PREFIX) or key_text.startswith(TSV_FFMETA_PREFIX):
                continue
            chapter[key_text.lower()] = _as_text((row or {}).get(key_text, ""))

        for key in list(header or []):
            key_text = str(key or "").strip()
            if (
                not key_text
                or key_text.startswith(TSV_META_PREFIX)
                or key_text.startswith(TSV_FFMETA_PREFIX)
                or key_text.lower() in chapter
            ):
                continue
            value = _as_text((row or {}).get(key_text, ""))
            if value != "":
                chapter[key_text.lower()] = value

        tb = _as_text(chapter.get("timebase", "")).strip()
        if tb:
            parsed_tb = _parse_timebase(tb)
            if parsed_tb is not None:
                chapter["timebase_num"], chapter["timebase_den"] = parsed_tb
        else:
            tb_num = _safe_int(chapter.get("timebase_num"))
            tb_den = _safe_int(chapter.get("timebase_den"))
            if tb_num is not None and tb_den not in (None, 0):
                chapter["timebase_num"] = int(tb_num)
                chapter["timebase_den"] = int(tb_den)

        start_raw = _safe_int(chapter.get("start_raw"))
        end_raw = _safe_int(chapter.get("end_raw"))
        if start_raw is None:
            start_raw = _safe_int(chapter.get("start"))
        if end_raw is None:
            end_raw = _safe_int(chapter.get("end"))
        if start_raw is None:
            start_raw = _safe_int(chapter.get("start_frame"))
        if end_raw is None:
            end_raw = _safe_int(chapter.get("end_frame"))
        if start_raw is not None:
            chapter["start_raw"] = int(start_raw)
        if end_raw is not None:
            chapter["end_raw"] = int(end_raw)

        if "timebase_num" not in chapter or "timebase_den" not in chapter:
            start_val = _as_text(chapter.get("start", "")).strip()
            end_val = _as_text(chapter.get("end", "")).strip()
            try:
                if start_val:
                    chapter["start_seconds"] = float(start_val)
                    chapter["start"] = float(start_val)
            except Exception:
                pass
            try:
                if end_val:
                    chapter["end_seconds"] = float(end_val)
                    chapter["end"] = float(end_val)
            except Exception:
                pass

        if not _as_text(chapter.get("title")).strip():
            alt = _as_text(chapter.get("chaptertitle") or chapter.get("chapter_title")).strip()
            if alt:
                chapter["title"] = alt

        chapters.append(chapter)

    return ffmeta, chapters


def _chapter_seconds(chapter: dict, boundary: str) -> float:
    raw_key = f"{boundary}_raw"
    try:
        raw = chapter.get(raw_key)
        tb_num = chapter.get("timebase_num")
        tb_den = chapter.get("timebase_den")
        if raw is not None and tb_num is not None and tb_den is not None:
            return float(Fraction(int(raw), 1) * Fraction(int(tb_num), int(tb_den)))
    except Exception:
        pass
    try:
        return float(chapter.get(f"{boundary}_seconds") or 0.0)
    except Exception:
        pass
    return float(chapter.get(boundary, 0.0) or 0.0)


def generate_tsv_metadata(chapters_tsv_path: Path, out_path: Path):
    ffmeta, chapters = _load_master_chapters(chapters_tsv_path)
    lines = ["Title\tAuthor\tChapterTitle\tStartSeconds\tEndSeconds\tLocation"]

    for chapter in chapters:
        start = round(_chapter_seconds(chapter, "start"), 3)
        end = round(_chapter_seconds(chapter, "end"), 3)

        lines.append(
            "\t".join(
                [
                    ffmeta.get("title", ""),
                    ffmeta.get("author", ffmeta.get("autor", "")),
                    chapter.get("title", ""),
                    str(start),
                    str(end),
                    chapter.get("location", ""),
                ]
            )
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Generated TSV metadata: {out_path}")


def generate_mkv_chapters_xml(chapters_tsv_path: Path, out_path: Path):
    _ffmeta, chapters = _load_master_chapters(chapters_tsv_path)

    def _fmt(seconds: float):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    root = ET.Element("Chapters")
    edition = ET.SubElement(root, "EditionEntry")

    for chapter in chapters:
        start = round(_chapter_seconds(chapter, "start"), 3)
        end = round(_chapter_seconds(chapter, "end"), 3)

        atom = ET.SubElement(edition, "ChapterAtom")
        ET.SubElement(atom, "ChapterTimeStart").text = _fmt(start)
        ET.SubElement(atom, "ChapterTimeEnd").text = _fmt(end)
        display = ET.SubElement(atom, "ChapterDisplay")
        ET.SubElement(display, "ChapterString").text = chapter.get("title", "") or ""
        ET.SubElement(display, "ChapterLanguage").text = "und"

    try:
        ET.indent(root, space="  ", level=0)
    except AttributeError:
        pass

    out_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE Chapters SYSTEM "matroskachapters.dtd">\n' + ET.tostring(root, encoding="unicode") + "\n",
        encoding="utf-8",
    )
    print(f"  Generated MKV chapters XML: {out_path}")


def write_mediainfo_outputs(input_path: Path, output_dir: Path):
    source = Path(input_path)
    outputs = [
        ("Text", f"{source.stem}_mediainfo.txt"),
        ("XML", f"{source.stem}_mediainfo.xml"),
    ]

    for fmt, filename in outputs:
        out_path = output_dir / filename
        cmd = [str(MEDIAINFO_BIN), f"--Output={fmt}", str(source)]
        try:
            with out_path.open("w", encoding="utf-8") as out:
                result = subprocess.run(cmd, cwd=output_dir, stdout=out, text=True)
                if result.returncode:
                    print(f"  ERROR: mediainfo {fmt} failed for {source}")
                    return int(result.returncode)
        except FileNotFoundError:
            print(f"  ERROR: mediainfo command not found: {MEDIAINFO_BIN}")
            print("  Install MediaInfo CLI (e.g. sudo apt-get install mediainfo) or set MEDIAINFO_BIN.")
            return 1
    return 0


def _generate_archive_metadata_for_store(root_dir: Path) -> int:
    files = sorted(
        glob.glob(str(root_dir / "*.mkv")) + glob.glob(str(root_dir / "*.flac")),
        key=lambda x: x.lower(),
    )
    if not files:
        return 0

    print(f"Processing directory: {root_dir.resolve()}")
    for fn in files:
        source = Path(fn)
        print(f"Processing: {source}")

        ad = archive_dir_for(source.stem)
        archive_metadata_dir = ad / f"{source.stem}_metadata"
        archive_metadata_dir.mkdir(exist_ok=True)
        rc = write_mediainfo_outputs(source, archive_metadata_dir)
        if rc:
            return rc

        chapters_tsv_path = METADATA_DIR / source.stem / "chapters.tsv"
        ffmetadata_path = METADATA_DIR / source.stem / "chapters.ffmetadata"
        tsv_path = METADATA_DIR / source.stem / "markers.tsv"
        mkv_chapter_path = METADATA_DIR / source.stem / "markers.mkvchapters.xml"
        if not chapters_tsv_path.exists() and ffmetadata_path.exists():
            ffmetadata_to_chapters_tsv(ffmetadata_path, chapters_tsv_path)

        if not chapters_tsv_path.exists():
            print(f"  Missing metadata file: {chapters_tsv_path}")
            return 1

        generate_ffmetadata_from_chapters_tsv(chapters_tsv_path, ffmetadata_path)
        generate_tsv_metadata(chapters_tsv_path, tsv_path)
        generate_mkv_chapters_xml(chapters_tsv_path, mkv_chapter_path)

        metadata_dir = ffmetadata_path.parent
        shutil.copytree(metadata_dir, archive_metadata_dir, dirs_exist_ok=True)

    checksum_file = archive_checksum_file_for(Path(files[0]).stem)
    write_sha3_manifest(root_dir, checksum_file, ignore_fn=lambda p: p.name == ".DS_Store")
    print(f"Checksum manifest: {checksum_file}")
    write_archive_readme(root_dir / "README.txt")
    print(f"Archive README: {root_dir / 'README.txt'}")
    return 0


def generate_archive_metadata():
    store_dirs = all_store_archive_dirs()
    if not store_dirs:
        print("No archive directories found.")
        return 1
    for root_dir in store_dirs:
        rc = _generate_archive_metadata_for_store(root_dir)
        if rc:
            return rc
    print("All done.")
    return 0


def _get_ffmpeg_version() -> str:
    try:
        result = subprocess.run(
            [str(FFMPEG_BIN), "-version"],
            capture_output=True,
            text=True,
        )
        return result.stdout.splitlines()[0].strip() if result.stdout else "unknown"
    except Exception:
        return "unknown"


def _get_whisper_version() -> str:
    try:
        return importlib.metadata.version("openai-whisper")
    except Exception:
        return "not installed"


def write_archive_readme(output_path: Path) -> None:
    ffmpeg_version = _get_ffmpeg_version()
    whisper_version = _get_whisper_version()

    whisper_prefilters = ", ".join(WHISPER_AUDIO_FILTERS)
    audio_channels_label = "mono" if ENCODE_AUDIO_CHANNELS == "1" else "stereo"

    content = f"""\
VHS DIGITIZATION AND PROCESSING PIPELINE
=========================================

STEP 1 — CAPTURE: VCR TO AVI
Hardware:
  VCR:          Panasonic AG-1970P
  Capture card: Osprey 260e
  Connection:   S-Video + unbalanced audio (L+R) from VCR A1 connectors

VCR settings:
  TBC: On
  Noise Filter: Off
  HiFi/NormalMix: Off
  Mono: On

Osprey 260e settings:
  Video input: S-Video
  Horizontal format: CCIR-601 (720px source width)
  Video standard: NTSC_M

Capture software: VirtualDub
  Codec: UT Video YUV422 BT.601 (UtVideo YUV422 BT.601.VCM)
    YUV 4:2:2 planar — luma (Y) sampled at full horizontal resolution,
    chroma (Cb/Cr) subsampled 2:1 horizontally, matching the analog VHS
    chroma bandwidth. BT.601 matrix coefficients for SD NTSC content.
  Frame rate: 29.97 fps
  Audio: PCM 48000 Hz, 16-bit, mono
  Synchronization: No correction (internal capture mode)
  Output format: AVI

---

STEP 2 — ARCHIVE: AVI TO MKV (lossless transcode)
Tool: {ffmpeg_version}
  ffmpeg -i input.avi \\
    -pix_fmt yuv422p \\
    -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v tv \\
    -c:v ffv1 -level 3 -g 1 -coder 1 -context 1 -slices 24 -slicecrc 1 \\
    -c:a pcm_s16le \\
    output.mkv

  Video: FFV1 level 3 (lossless), every frame a keyframe (g=1)
  Pixel format: YUV 4:2:2 planar (yuv422p) — preserves the full chroma
    resolution of the capture; no chroma decimation at the archive stage
  Color metadata tags:
    color_primaries 6 = SMPTE 170M (standard NTSC primaries)
    color_trc 6       = SMPTE 170M (BT.601 gamma transfer function)
    colorspace 5      = BT.470BG / SMPTE 170M (SD YCbCr matrix)
    color_range tv    = limited range (16-235 luma, 16-240 chroma)
  Audio: PCM signed 16-bit LE passthrough (no re-encode)
  Chapter metadata embedded if available
  Container: Matroska (.mkv)

---

STEP 3 — CHAPTER MARKING
Chapter start and end boundaries are identified by reviewing the archive
MKV in a video player and noting frame numbers at scene transitions.
Timecode conversion: seconds = frame_number / (30000/1001)  [NTSC 29.97fps]
Boundaries are stored as an FFmpeg ffmetadata file alongside the archive.

---

STEP 4 — CHAPTER EXTRACTION AND DEINTERLACING
Each chapter is extracted from the archive MKV and deinterlaced individually.

Extraction: ffmpeg with per-chapter start/end frame range
Deinterlacing: AviSynth+ / QTGMC (FFmpeg-QTGMC Easy 2025.01.11)
  Parameters vary per archive — see filter.avs in each archive's metadata directory.
  Typical settings: Preset="Very Slow", SourceMatch=3, Lossless=2, TR2=3
  FPSDivisor=2: field-matched single-rate output (29.97fps in -> 29.97fps out)
  SourceMatch=3: highest quality motion-compensated field matching
  Lossless=2: lossless refinement pass after deinterlacing
Intermediate output: FFV1 lossless MKV, YUV 4:2:2 planar (not retained after final encode)

Bad frame handling:
  Tape dropout and tracking loss artifacts are detected automatically by
  scoring every frame against four computer vision signals:
    - chroma_loss:   1 - mean(HSV saturation)/255 — desaturated/grey frames
                     caused by chroma dropout score high
    - noise_energy:  std(per-row variance) / mean(per-row variance) — tracking
                     noise creates a few rows with anomalously high variance,
                     spiking this signal even when only a small portion of the
                     frame is affected
    - row_tear:      95th-percentile of row-to-neighbour mean absolute
                     difference — horizontal tearing displaces rows, causing
                     large inter-row differences
    - wave_energy:   std of the high-passed per-row luminance centre-of-mass —
                     horizontal sync instability manifests as a left/right
                     oscillation of row content; the CoM series is high-passed
                     (5-row rolling mean subtracted) to isolate rapid sync
                     wobble from slow scene content gradients

  The four signals are weighted equally (0.25 each) and combined into a
  single composite score per frame. Thresholds are computed per chapter using
  a Tukey fence: threshold = Q3 + 3.5 * IQR, evaluated within sliding
  1000-frame windows aligned to chapter boundaries. Frames whose composite
  score exceeds their chapter window's threshold are flagged as candidates.

  Flagged frames are then manually reviewed by inspecting each candidate
  visually to confirm genuine damage. Confirmed bad frame numbers are
  recorded per-archive alongside the filter script. During rendering, the
  AviSynth script replaces each confirmed bad frame with a copy of the
  nearest clean frame using FreezeFrame(). If a large number of consecutive
  frames required replacement, the result will appear as a brief freeze in
  the output video. This is intentional — it is preferable to a corrupted or
  visually broken frame.

---

STEP 5 — AUDIO TRANSCRIPTION (chapters with transcript enabled)
Tool: OpenAI Whisper, model: {WHISPER_MODEL}
Installed version: {whisper_version}
Audio pre-processing chain: {whisper_prefilters}
  HPF 120 Hz (6 dB/oct Butterworth) — attenuates low-frequency mechanical noise and hum
  LPF 8 kHz — removes HF noise and aliasing above the speech intelligibility ceiling
  afftdn nf=-25 dB — STFT-domain noise reduction; noise floor estimated from signal
  dynaudnorm f=150 ms / g=13 — frame-level RMS gain normalization with Gaussian smoothing
  aresample 16000 Hz — downsample to Whisper's expected input sample rate
  loudnorm I=-16 LUFS / TP=-1.5 dBTP / LRA=11 LU — EBU R128 integrated loudness normalization
Output: PCM s16le, 16000 Hz, mono
Subtitle sidecar formats produced: .srt, .vtt, .ass

---

STEP 6 — FINAL ENCODE: DEINTERLACED CHAPTER TO MP4
Tool: {ffmpeg_version}

Video:
  Codec: {ENCODE_VIDEO_CODEC}, preset {ENCODE_VIDEO_PRESET}, CRF {ENCODE_VIDEO_CRF}
  Profile: {ENCODE_VIDEO_PROFILE.capitalize()}, Level {ENCODE_VIDEO_LEVEL}, tune {ENCODE_VIDEO_TUNE}
  Pixel format: YUV 4:2:0 planar ({ENCODE_VIDEO_PIX_FMT}) — chroma subsampled
    2:1 horizontally and vertically for H.264 compatibility; converted from
    the 4:2:2 intermediate at the final encode stage
  FPS passthrough (no rate conversion)

Audio:
  Codec: {ENCODE_AUDIO_CODEC.upper()} {ENCODE_AUDIO_BITRATE}, {audio_channels_label}, {ENCODE_AUDIO_SAMPLE_RATE} Hz
  Processing chain: {ENCODE_AUDIO_FILTERS}
  HPF 80 Hz — rolls off sub-bass rumble and low-frequency tape noise
  LPF 14 kHz — removes HF tape hiss above the audible program bandwidth
  afftdn nf=-25 dB — STFT-domain noise reduction
  loudnorm I=-16 LUFS / TP=-1.5 dBTP / LRA=11 LU — EBU R128 integrated loudness normalization

Subtitles: ASS tracks embedded (dialogue and/or named people)
Metadata embedded: title, author, creation_time, location
Container: MP4

---

SIDECAR AND METADATA FILES (per archive, in {{archive}}_metadata/)
  {{stem}}_mediainfo.txt    MediaInfo technical track analysis (text)
  {{stem}}_mediainfo.xml    MediaInfo technical track analysis (XML)
  chapters.ffmetadata       FFmpeg chapter/global metadata
  chapters.tsv              Editable master chapter table (title, timecodes, location, etc.)
  markers.tsv               Flat export: Title, Author, ChapterTitle, StartSeconds, EndSeconds, Location
  markers.mkvchapters.xml   Matroska XML chapter format for muxing tools
  filter.avs                AviSynth script used for QTGMC deinterlacing and bad frame repair
  subtitles.tsv             Curated dialogue subtitle entries with confidence scores
  people.tsv                Named person time-range entries for people subtitle track
  comment.txt               Free-text archival notes embedded in archive outputs
  SHA256SUMS                SHA-3/256 checksum manifest for all archive files
  README.txt                This file
"""
    Path(output_path).write_text(content, encoding="utf-8")


def main(argv=None):
    _ = argv
    return generate_archive_metadata()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
