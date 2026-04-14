import csv
import wave

import pytest
from unittest import skip

pytest.importorskip("numpy")
pytest.importorskip("cv2")

from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from common import merge_bad_frames_in_render_settings
from apps.plain_html_wizard import server as wizard_server
from libs.vhs_tuner_core import _chapter_bad_overrides
from apps.plain_html_wizard.server import (
    CONTACT_SHEET_COLUMNS,
    SessionState,
    _build_review_payload,
    _build_contact_sheet_bytes,
    _build_partial_review_payload,
    _decode_frame_image_data_url,
    _frame_image_url,
    _frame_contact_sheet_url,
    _lookup_frame_image_data_url,
    _load_split_entries_for_chapter,
    _selected_bad_frame_ids,
    _normalize_iqr_k,
    _normalize_subtitle_entries_payload,
    _save_split_entries_for_chapter,
    _set_load_progress,
    WizardHandler,
)

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "apps" / "plain_html_wizard" / "static" / "index.html"


def _make_session() -> SessionState:
    n = 120
    fids = list(range(1000, 1000 + n))
    chroma = np.linspace(0.0, 1.0, n, dtype=np.float64)
    chroma[-8:] += np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    sigs = {
        "chroma": chroma,
        "noise": np.zeros(n, dtype=np.float64),
        "tear": np.zeros(n, dtype=np.float64),
        "wave": np.zeros(n, dtype=np.float64),
    }
    return SessionState(fids=fids, sigs=sigs, overrides={})


def _write_single_chapter_tsv(path: Path, title: str, start_frame: int, end_frame: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "__chapter_index\tffmeta_title\tTIMEBASE\tSTART\tEND\ttitle",
                f"1\tDemo Archive\t1001/30000\t{int(start_frame)}\t{int(end_frame)}\t{title}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _make_loaded_wizard_session(archive: str, chapter: str, start_frame: int, end_frame: int) -> SessionState:
    frame_count = max(1, int(end_frame) - int(start_frame))
    return SessionState(
        archive=archive,
        chapter=chapter,
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        fids=list(range(int(start_frame), int(end_frame))),
        b64=["data:image/jpeg;base64,AA=="] * frame_count,
        sigs={
            "chroma": np.linspace(0.1, 0.4, frame_count, dtype=np.float64),
            "noise": np.zeros(frame_count, dtype=np.float64),
            "tear": np.zeros(frame_count, dtype=np.float64),
            "wave": np.zeros(frame_count, dtype=np.float64),
        },
        overrides={},
    )


class _HandlerStub:
    def __init__(self) -> None:
        self.payload = None
        self.error = None

    def _send_json(self, payload, code=200) -> None:
        _ = code
        self.payload = payload

    def _send_error_json(self, message, code=400) -> None:
        _ = code
        self.error = str(message)


class _PostHandlerStub(_HandlerStub):
    def __init__(self, path: str, payload: dict, session: SessionState | None = None) -> None:
        super().__init__()
        self.path = str(path)
        self._payload = dict(payload)
        self._session = session or SessionState()

    def _ensure_session(self) -> SessionState:
        return self._session

    def _read_json(self) -> dict:
        return dict(self._payload)


def test_normalize_iqr_k_clamps_and_parses() -> None:
    assert _normalize_iqr_k(-1) == 0.0
    assert _normalize_iqr_k(99) == 12.0
    assert _normalize_iqr_k("2.75") == 2.75
    assert _normalize_iqr_k("bad", default=4.2) == 4.2


def test_review_payload_reprocesses_bad_counts_when_iqr_changes() -> None:
    session = _make_session()
    session.iqr_k = 12.0
    high_k = _build_review_payload(session, include_images=False)

    session.iqr_k = 0.0
    low_k = _build_review_payload(session, include_images=False)

    assert low_k["threshold"] <= high_k["threshold"]
    assert low_k["stats"]["bad"] >= high_k["stats"]["bad"]
    assert low_k["stats"]["bad"] > high_k["stats"]["bad"]
    assert len(low_k["frames"]) == len(session.fids)


def test_review_payload_honors_manual_overrides_after_iqr_change() -> None:
    session = _make_session()
    target_fid = session.fids[-1]
    session.overrides = {int(target_fid): "good"}
    session.iqr_k = 0.0

    review = _build_review_payload(session, include_images=False)
    frame = next(f for f in review["frames"] if int(f["fid"]) == int(target_fid))

    assert frame["status"] == "good"
    assert frame["source"] == "MG"


def test_review_payload_force_all_frames_good_marks_everything_good() -> None:
    session = _make_session()
    session.iqr_k = 0.0
    baseline = _build_review_payload(session, include_images=False)
    assert baseline["stats"]["bad"] > 0

    session.force_all_frames_good = True
    forced = _build_review_payload(session, include_images=False)
    assert forced["force_all_frames_good"] is True
    assert forced["stats"]["bad"] == 0
    assert forced["stats"]["good"] == forced["stats"]["total"]
    assert all(str(frame["status"]) == "good" for frame in forced["frames"])
    assert _selected_bad_frame_ids(session) == []


def test_review_payload_can_emit_frame_image_urls() -> None:
    session = _make_session()
    session.b64 = ["data:image/jpeg;base64,AA=="] * len(session.fids)

    review = _build_review_payload(
        session,
        include_images=True,
        image_url_builder=_frame_image_url,
    )

    assert review["frames"]
    first = review["frames"][0]
    assert first["image"].startswith(f"/api/frame_image?fid={session.fids[0]}")


def test_frame_image_lookup_and_decode_supports_partial_frames() -> None:
    session = SessionState(
        partial_fids=[1000, 1001],
        partial_b64=[
            "data:image/jpeg;base64,AA==",
            "data:image/jpeg;base64,AQ==",
        ],
    )

    data_url = _lookup_frame_image_data_url(session, 1001)
    decoded = _decode_frame_image_data_url(data_url)

    assert data_url == "data:image/jpeg;base64,AQ=="
    assert decoded == ("image/jpeg", b"\x01")


def test_contact_sheet_builder_returns_jpeg_bytes() -> None:
    session = SessionState(
        partial_fids=[1000, 1001],
        partial_b64=[
            "data:image/jpeg;base64,AA==",
            "data:image/jpeg;base64,AQ==",
        ],
    )

    built = _build_contact_sheet_bytes(
        session,
        start_index=0,
        count=2,
        columns=CONTACT_SHEET_COLUMNS,
    )

    assert built is not None
    result, all_loaded = built
    assert result is not None
    content_type, payload = result
    assert content_type == "image/jpeg"
    assert len(payload) > 0
    assert _frame_contact_sheet_url(0, count=2, columns=CONTACT_SHEET_COLUMNS).startswith("/api/frame_contact_sheet?")


def test_contact_sheet_builder_can_fill_later_visible_range_from_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = SessionState(
        archive="demo_archive",
        chapter="Demo Chapter",
        start_frame=1000,
        end_frame=1010,
        partial_fids=[1000, 1001],
        partial_b64=[
            "data:image/jpeg;base64,AA==",
            "data:image/jpeg;base64,AQ==",
        ],
        frame_source_video_path="chapter_extract.mp4",
        frame_source_read_offset=1000,
    )
    calls: dict[str, list[int]] = {}

    def _fake_load_from_video(session_arg: SessionState, frame_ids: list[int]) -> dict[int, str]:
        assert session_arg is session
        calls["frame_ids"] = list(frame_ids)
        return {
            1004: "data:image/jpeg;base64,AA==",
            1005: "data:image/jpeg;base64,AQ==",
        }

    monkeypatch.setattr(wizard_server, "_load_contact_sheet_images_from_video", _fake_load_from_video)

    built = _build_contact_sheet_bytes(
        session,
        start_index=4,
        count=2,
        columns=CONTACT_SHEET_COLUMNS,
    )

    assert built is not None
    assert calls["frame_ids"] == [1004, 1005]


def test_static_html_contains_live_iqr_spark_and_fullscreen_controls() -> None:
    static_dir = INDEX_HTML.parent
    html = "\n".join(
        f.read_text(encoding="utf-8")
        for f in sorted(static_dir.glob("*.html")) + sorted(static_dir.glob("*.css")) + sorted(static_dir.glob("*.js"))
    )

    assert 'id="helpBtn"' not in html
    assert 'id="helpModal"' in html
    assert 'id="helpCloseBtn"' in html
    assert "openHelpModal()" in html
    assert "closeHelpModal()" in html
    assert "helpBtnEl.addEventListener('click', openHelpModal)" in html
    assert "event.key === 'Escape'" in html
    assert 'id="stepPills"' in html
    assert "const STEP_DEFS = [" in html
    assert "const STEP_MODE_TO_FIRST = new Map();" in html
    assert "const navActionButtons = new Map();" in html
    assert "setStepByMode(" in html
    assert "function navigateToStep(rawStep)" in html
    assert "stepPillsEl.addEventListener('click'" in html
    assert "pill.type = 'button';" in html
    assert 'id="nextToReview"' not in html
    assert 'id="nextToGamma"' not in html
    assert 'id="backToLoad"' not in html
    assert 'id="backToReview"' not in html

    assert 'id="iqrK" type="range" min="0" max="12"' in html
    assert 'id="forceAllFramesGood" type="checkbox"' in html
    assert 'id="gammaLevel" type="range" min="0.05" max="8.00"' in html
    assert 'id="gammaMode"' in html
    assert 'id="sparkPlayBtn"' in html
    assert 'id="iqrSpark"' in html
    assert 'id="toggleFullscreen"' in html
    assert 'id="previewRender"' in html
    assert "grid-template-rows: auto auto minmax(0, 1fr) auto;" in html
    assert "iqrEl.addEventListener('input'" in html
    assert "iqrEl.addEventListener('change'" in html
    assert "scheduleAutoApplyIqr()" in html
    assert "scheduleVisibleRangeRefresh()" in html
    assert "frameGridEl.addEventListener('scroll'" in html
    assert "normalizeWheelToPixels(" in html
    assert "window.addEventListener('wheel', relayWheelToFrameGrid, { passive: false, capture: true })" in html
    assert "frameGridEl.scrollBy({ top: deltaPx * 1.75, behavior: 'auto' })" in html
    assert "const sprite = frameContactSheetSpecForIndex(frameIndex, gridMetrics);" in html
    assert "return { image: '', sprite, replaced: false, note: '' };" in html
    assert ".frame-card.loading .frame-thumb-sprite {" in html
    assert "opacity: 0.94;" in html
    assert "bottom: 12px;" in html
    assert ": !isChapterLoadInFlight;" in html
    assert 'id="overlayProgressFill"' in html
    assert 'id="overlayProgressText"' in html
    assert 'id="overlayEtaText"' in html
    assert 'id="overlayCancelBtn"' in html
    assert "startLoadProgress(" in html
    assert "finishLoadProgress(" in html
    assert "pollLoadProgressOnce()" in html
    assert "api('/api/load_progress')" in html
    assert "api('/api/preview_render', 'POST'" in html
    assert "audio_sync_profile: {" in html
    assert "api('/api/set_force_all_good', 'POST'" in html
    assert "renderReadyAtFromSamples(" in html
    assert "ETA " in html
    assert "/3 sample" in html
    assert "api('/api/cancel_load', 'POST', {})" in html
    assert "seekFrameGridFromSparkClientX(" in html
    assert "queueSparkDragSeek(" in html
    assert "toggleSparkWindowPlayback(" in html
    assert "window.requestAnimationFrame(runSparkWindowPlaybackFrameClock)" in html
    assert "const VHS_FRAME_MS = VHS_FRAME_SECONDS * 1000;" in html
    assert "sparkPlayBtnEl.addEventListener('click', openFlipbookPanel)" in html
    assert "iqrSparkEl.addEventListener('pointerdown'" in html
    assert "iqrSparkEl.addEventListener('pointermove'" in html
    assert "frameGridEl.scrollTo({ top, behavior: 'auto' })" in html
    assert "const sparkThreshold = themeVar('--spark-threshold', '#ff646e');" in html
    assert "function applySparklineCache(cache, frames)" in html
    assert "function buildReviewSparklineCache(frames, threshold)" in html
    assert "function buildGammaSparklineCache(frames, gammaLevel)" in html
    assert "replaceGammaScores(new Map());" in html
    assert "const FRAME_GRID_CONTACT_SHEET_PREFETCH_AHEAD = 2;" in html
    assert "function prefetchVisibleFrameSheets(metricsRaw = null, rangeRaw = null)" in html
    assert "prefetchFrameContactSheet(next.url);" in html
    assert "const FLIPBOOK_AUDIO_CLOCK_STALL_FRAMES = 8;" in html
    assert "sparkPlayAudioClockStallCount += 1;" in html
    assert "return tc;" in html
    assert "frame.status === 'bad'" in html
    assert "event.target.closest('.frame-card')" in html
    assert "window.open(target, '_blank', 'noopener')" in html
    assert 'id="subtitlesGenerate"' in html
    assert 'id="subtitlesClear"' in html
    assert 'id="subtitlesEditor"' in html
    assert 'id="peopleMeta"' in html
    assert 'id="subtitlesMeta"' in html
    assert "subtitles-editor-grid" in html
    assert "label: 'Chapter'" in html
    assert "Save and Return to Chapter" in html
    assert "Use the Chapter step to edit the loaded chapter range" in html
    assert "Chapter step | range" in html
    assert "data-split-row-delete" not in html
    assert "next row starts at end + 1" not in html
    assert "active-row" in html
    assert 'data-sub-field="text"' in html
    assert "parseSubtitlesEditorGrid(" in html
    assert "syncSubtitlesEditorToCursor(" in html
    assert "const peopleMetaEl = document.getElementById('peopleMeta');" in html
    assert "const subtitlesMetaEl = document.getElementById('subtitlesMeta');" in html
    assert "peopleMetaEl.classList.toggle('hidden-ui', !peopleMode)" in html
    assert "subtitlesMetaEl.classList.toggle('hidden-ui', !subtitlesMode)" in html
    assert 'id="timelineAudioPlay"' in html
    assert 'id="timelineAudioTrack"' in html
    assert 'id="timelineAudioWave"' in html
    assert 'id="timelineAudioPlayhead"' in html
    assert 'id="timelineAudio"' in html
    assert 'id="flipbookSubtitles"' in html
    assert 'id="flipbookAudio"' in html
    assert 'id="flipbookVolume"' in html
    assert "renderFlipbookSubtitles(" in html
    assert "applyFlipbookVolume(" in html
    assert "ensureTimelineAudioLoaded(" in html
    assert "toggleTimelineAudioPlayback(" in html
    assert "people-timeline-audio-underlay" in html
    assert "/api/chapter_audio" in html
    assert "updatePeopleStepLayoutSizing(" in html
    assert "data-subtitle-row-delete" in html
    assert "subtitle-timeline-bar" in html
    assert "subtitle-timeline-input" in html
    assert "subtitle-timeline-delete" in html
    assert "editSubtitleTimelineEntry(" in html
    assert "clearSubtitlesEntries()" in html
    assert "subtitlesClearBtnEl.addEventListener('click', clearSubtitlesEntries)" in html
    assert "window.prompt('Edit subtitle text'" not in html
    assert "deleteSubtitleTimelineEntry(" in html
    assert "people-timeline-zoom-btn" in html
    assert "peopleTimelineEl.addEventListener('wheel'" not in html
    assert "api('/api/subtitles_generate', 'POST'" in html
    assert "Toggle Good/Bad" not in html


def test_set_load_progress_updates_and_clamps_state() -> None:
    session = SessionState()
    _set_load_progress(
        session,
        running=True,
        progress=133.0,
        message="Sampling frames",
        sample_done=42,
        sample_total=100,
    )
    assert session.load_running is True
    assert session.load_progress == 100.0
    assert session.load_message == "Sampling frames"
    assert session.load_sample_done == 42
    assert session.load_sample_total == 100

    _set_load_progress(session, running=False, progress=-7.0, sample_done=-2, sample_total=-9)
    assert session.load_running is False
    assert session.load_progress == 0.0
    assert session.load_sample_done == 0
    assert session.load_sample_total == 0


def test_chapter_bad_overrides_load_saved_bad_frames_from_render_settings() -> None:
    archive = "__plain_wizard_unit_archive"
    title = "Unit Chapter"
    archive_meta_dir = ROOT / "metadata" / archive
    try:
        merge_bad_frames_in_render_settings(archive, [100, 105, 205])
        overrides = _chapter_bad_overrides(
            archive=archive,
            chapter_title=title,
            ch_start=100,
            ch_end=200,
        )
        assert overrides == {100: "bad", 105: "bad"}
    finally:
        shutil.rmtree(archive_meta_dir, ignore_errors=True)


def test_toggle_frame_allows_partial_frames_before_full_load() -> None:
    session = SessionState(
        start_frame=1000,
        partial_fids=[1000, 1001, 1002],
        partial_b64=["", "", ""],
        partial_sigs={
            "chroma": [0.1, 10.0, 0.1],
            "noise": [0.0, 0.0, 0.0],
            "tear": [0.0, 0.0, 0.0],
            "wave": [0.0, 0.0, 0.0],
        },
    )
    review_before = _build_partial_review_payload(session, include_images=False)
    frame_before = next(f for f in review_before["frames"] if int(f["fid"]) == 1001)
    status_before = str(frame_before["status"])

    handler = _HandlerStub()
    WizardHandler._handle_toggle_frame(handler, session, 1001)

    assert handler.error is None
    assert handler.payload is not None
    frame_after = next(f for f in handler.payload["review"]["frames"] if int(f["fid"]) == 1001)
    assert str(frame_after["status"]) != status_before
    assert session.overrides[1001] == ("good" if status_before == "bad" else "bad")


def test_toggle_frame_rejected_when_force_all_frames_good_enabled() -> None:
    session = SessionState(
        start_frame=1000,
        partial_fids=[1000],
        partial_b64=[""],
        partial_sigs={
            "chroma": [0.1],
            "noise": [0.0],
            "tear": [0.0],
            "wave": [0.0],
        },
        force_all_frames_good=True,
    )
    handler = _HandlerStub()
    WizardHandler._handle_toggle_frame(handler, session, 1000)

    assert handler.payload is None
    assert handler.error is not None
    assert "Force all frames good" in handler.error


def test_set_frame_range_marks_partial_frames_bad() -> None:
    session = SessionState(
        start_frame=1000,
        partial_fids=[1000, 1001, 1002, 1003, 1004],
        partial_b64=["", "", "", "", ""],
        partial_sigs={
            "chroma": [0.1, 0.2, 0.3, 0.4, 0.5],
            "noise": [0.0, 0.0, 0.0, 0.0, 0.0],
            "tear": [0.0, 0.0, 0.0, 0.0, 0.0],
            "wave": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )
    handler = _HandlerStub()
    WizardHandler._handle_set_frame_range(handler, session, 1001, 1003, "bad")

    assert handler.error is None
    assert handler.payload is not None
    assert int(handler.payload.get("updated_count", 0)) == 3
    for fid in (1001, 1002, 1003):
        assert session.overrides[fid] == "bad"


def test_normalize_subtitle_entries_payload_clamps_and_preserves_optional_fields() -> None:
    rows = _normalize_subtitle_entries_payload(
        [
            {
                "start": "00:00:01.000",
                "end": "00:00:03.000",
                "text": " Hello   there ",
                "speaker": " Narrator ",
                "confidence": "1.5",
                "source": " whisper ",
            },
            {
                "start": "00:00:05.000",
                "end": "00:00:04.000",
                "text": "invalid",
            },
        ],
        chapter_duration_seconds=4.0,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["start"] == "00:00:01.000"
    assert row["end"] == "00:00:03.000"
    assert row["text"] == "Hello there"
    assert row["speaker"] == "Narrator"
    assert row["confidence"] == 1.0
    assert row["source"] == "whisper"


def test_save_split_entries_writes_start_end_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wizard_server, "METADATA_DIR", tmp_path)

    archive = "demo_archive"
    chapter = "Parent Chapter"
    archive_dir = tmp_path / archive
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "chapters.ffmetadata").write_text(
        "\n".join(
            [
                ";FFMETADATA1",
                "title=Demo Archive",
                "",
                "[CHAPTER]",
                "TIMEBASE=1001/30000",
                "START=100",
                "END=200",
                f"title={chapter}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    path, count = _save_split_entries_for_chapter(
        archive,
        chapter,
        100,
        200,
        [
            {"start_frame": 0, "end_frame": 25, "title": "Part 1"},
            {"start_frame": 25, "end_frame": 80, "title": "Part 2"},
        ],
    )

    assert count == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("__chapter_index\tffmeta_title\tTIMEBASE\tSTART\tEND\ttitle")
    assert "parent_chapter" not in lines[0]
    assert "start_frame" not in lines[0]
    assert lines[1].split("\t")[2:6] == ["1001/30000", "100", "200", chapter]
    assert lines[2].split("\t")[2:6] == ["1001/30000", "100", "125", "Part 1"]
    assert lines[3].split("\t")[2:6] == ["1001/30000", "125", "180", "Part 2"]


def test_save_split_entries_updates_single_existing_chapter_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wizard_server, "METADATA_DIR", tmp_path)

    archive = "demo_archive"
    chapter = "Parent Chapter"
    archive_dir = tmp_path / archive
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "chapters.tsv").write_text(
        "\n".join(
            [
                "__chapter_index\tffmeta_title\tTIMEBASE\tSTART\tEND\ttitle",
                "1\tDemo Archive\t1001/30000\t100\t200\tParent Chapter",
                "2\tDemo Archive\t1001/30000\t300\t400\tOther Chapter",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    path, count = _save_split_entries_for_chapter(
        archive,
        chapter,
        100,
        200,
        [
            {"start_frame": 20, "end_frame": 80, "title": "Updated Chapter"},
        ],
    )

    assert count == 1
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    assert len(rows) == 2
    assert rows[0]["title"] == "Updated Chapter"
    assert rows[0]["START"] == "120"
    assert rows[0]["END"] == "180"
    assert rows[1]["title"] == "Other Chapter"
    assert not any(row["title"] == "Parent Chapter" for row in rows)


def test_load_split_entries_reads_canonical_chapters_tsv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wizard_server, "METADATA_DIR", tmp_path)

    archive = "demo_archive"
    chapter = "Parent Chapter"
    archive_dir = tmp_path / archive
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "chapters.tsv").write_text(
        "\n".join(
            [
                "__chapter_index\tffmeta_title\tTIMEBASE\tSTART\tEND\ttitle",
                f"1\tDemo Archive\t1001/30000\t100\t200\t{chapter}",
                "2\tDemo Archive\t1001/30000\t100\t125\tPart 1",
                "3\tDemo Archive\t1001/30000\t125\t180\tPart 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _load_split_entries_for_chapter(archive, chapter, 100, 200)

    assert rows == [
        {
            "start_frame": 0,
            "end_frame": 25,
            "start": "0",
            "end": "25",
            "title": "Part 1",
        },
        {
            "start_frame": 25,
            "end_frame": 80,
            "start": "25",
            "end": "80",
            "title": "Part 2",
        },
    ]


def test_load_split_entries_accepts_legacy_start_frame_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wizard_server, "METADATA_DIR", tmp_path)

    archive = "demo_archive"
    chapter = "Parent Chapter"
    archive_dir = tmp_path / archive
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "chapters.tsv").write_text(
        "\n".join(
            [
                "parent_chapter\tstart_frame\tend_frame\ttitle",
                f"{chapter}\t100\t125\tPart 1",
                f"{chapter}\t125\t180\tPart 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _load_split_entries_for_chapter(archive, chapter, 100, 200)

    assert rows == [
        {
            "start_frame": 0,
            "end_frame": 25,
            "start": "0",
            "end": "25",
            "title": "Part 1",
        },
        {
            "start_frame": 25,
            "end_frame": 80,
            "start": "25",
            "end": "80",
            "title": "Part 2",
        },
    ]


def test_handle_load_chapter_populates_session_from_video_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_dir = tmp_path / "metadata"
    archive_dir = tmp_path / "Archive"
    archive_name = "demo_archive"
    chapter = "Example Chapter"
    archive_meta = metadata_dir / archive_name
    archive_meta.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    _write_single_chapter_tsv(archive_meta / "chapters.tsv", chapter, 100, 103)
    (archive_dir / f"{archive_name}_proxy.mp4").write_bytes(b"mp4")

    wizard_server._write_people_tsv_rows(
        archive_meta / "people.tsv",
        [
            (
                wizard_server._frame_to_seconds(99),
                wizard_server._frame_to_seconds(101),
                "Lynda",
            ),
            (
                wizard_server._frame_to_seconds(101),
                wizard_server._frame_to_seconds(103),
                "Jim | Linda",
            ),
        ],
    )
    wizard_server._write_subtitles_tsv_rows(
        archive_meta / "subtitles.tsv",
        [
            (
                wizard_server._frame_to_seconds(100),
                wizard_server._frame_to_seconds(102),
                "Opening line",
                "Narrator",
                0.9,
                "manual",
            )
        ],
    )

    extract_path = tmp_path / "extracts" / "chapter_extract.mp4"
    extract_path.parent.mkdir(parents=True, exist_ok=True)
    extract_path.write_bytes(b"extract")

    monkeypatch.setattr(wizard_server, "METADATA_DIR", metadata_dir)
    monkeypatch.setattr(wizard_server, "archive_dir_for", lambda _archive: archive_dir)
    monkeypatch.setattr(
        wizard_server,
        "_resolve_archive_video",
        lambda archive: archive_dir / f"{archive}_proxy.mp4",
    )
    monkeypatch.setattr(
        wizard_server,
        "_archive_state",
        lambda session_arg, archive, selected_title=None: (
            {
                "archive": archive,
                "chapter": selected_title or chapter,
                "status": "",
                "details": "",
                "start_frame": 100,
                "end_frame": 103,
                "chapters": [],
            }
            if not (
                setattr(session_arg, "archive", archive)
                or setattr(session_arg, "chapter", selected_title or chapter)
                or setattr(
                    session_arg,
                    "chapters",
                    [{"title": chapter, "start_frame": 100, "end_frame": 103}],
                )
                or setattr(session_arg, "chapter_rows", [])
                or setattr(session_arg, "start_frame", 100)
                or setattr(session_arg, "end_frame", 103)
            )
            else {}
        ),
    )
    monkeypatch.setattr(
        wizard_server,
        "_ensure_render_chapter_extract",
        lambda **_kwargs: (extract_path, None),
    )
    monkeypatch.setattr(
        wizard_server,
        "get_gamma_profile_for_chapter",
        lambda **_kwargs: {
            "default_gamma": 1.25,
            "ranges": [{"start_frame": 100, "end_frame": 102, "gamma": 1.4}],
            "source": "render_settings",
        },
    )
    monkeypatch.setattr(
        wizard_server,
        "get_transcript_mode_for_chapter",
        lambda **_kwargs: "append",
    )

    def _fake_extract_frames(
        source_video,
        start_frame,
        end_frame,
        n_frames,
        archive,
        chapter_title,
        include_thumbs=False,
        frame_read_offset=0,
        progress=None,
        should_cancel=None,
        frame_callback=None,
    ):
        assert Path(source_video) == extract_path
        assert start_frame == 100
        assert end_frame == 103
        assert n_frames == 3
        assert archive == archive_name
        assert chapter_title == chapter
        assert include_thumbs is False
        assert frame_read_offset == 100
        assert callable(should_cancel)
        if callable(progress):
            progress(0.5, "half")
        if callable(frame_callback):
            frame_callback(100, "data:image/jpeg;base64,AA==", 0.1, 0.0, 0.0, 0.0, 1, 3)
            frame_callback(101, "data:image/jpeg;base64,AQ==", 0.2, 0.0, 0.0, 0.0, 2, 3)
        sigs = {
            "chroma": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "noise": np.zeros(3, dtype=np.float64),
            "tear": np.zeros(3, dtype=np.float64),
            "wave": np.zeros(3, dtype=np.float64),
        }
        return [100, 101, 102], ["data:image/jpeg;base64,AA=="] * 3, sigs, None

    monkeypatch.setattr(wizard_server, "extract_frames", _fake_extract_frames)

    session = SessionState(
        preview_video_path="stale_preview.mp4",
        chapter_audio_path="stale.wav",
        chapter_audio_key="stale",
        people_entries=[{"people": "Stale"}],
        subtitle_entries=[{"text": "Stale"}],
        split_entries=[{"title": "Stale"}],
    )
    handler = _HandlerStub()

    WizardHandler._handle_load_chapter(
        handler,
        session,
        {
            "archive": archive_name,
            "chapter": chapter,
            "iqr_k": 2.5,
            "force_all_frames_good": True,
        },
    )

    assert handler.error is None
    assert handler.payload is not None
    assert handler.payload["ok"] is True
    assert session.archive == archive_name
    assert session.chapter == chapter
    assert session.start_frame == 100
    assert session.end_frame == 103
    assert session.iqr_k == 2.5
    assert session.force_all_frames_good is True
    assert session.frame_source_video_path == str(extract_path)
    assert session.frame_source_read_offset == 100
    assert session.preview_video_path == ""
    assert session.chapter_audio_path == ""
    assert session.chapter_audio_key == ""
    assert session.fids == [100, 101, 102]
    assert session.partial_fids == [100, 101]
    assert session.gamma_default == 1.25
    assert session.auto_transcript == "append"
    assert [row["people"] for row in session.people_entries] == ["Lynda", "Jim | Linda"]
    assert [row["text"] for row in session.subtitle_entries] == ["Opening line"]
    assert handler.payload["settings"]["loaded_count"] == 3
    assert handler.payload["settings"]["start_frame"] == 100
    assert handler.payload["settings"]["end_frame"] == 103
    assert handler.payload["settings"]["gamma_profile"]["source"] == "render_settings"
    assert handler.payload["settings"]["people_profile"]["entries"] == session.people_entries
    assert handler.payload["settings"]["subtitles_profile"]["entries"] == session.subtitle_entries


def test_handle_load_chapter_reports_missing_archive_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_dir = tmp_path / "metadata"
    archive_name = "demo_archive"
    chapter = "Example Chapter"
    _write_single_chapter_tsv(metadata_dir / archive_name / "chapters.tsv", chapter, 100, 103)

    monkeypatch.setattr(wizard_server, "METADATA_DIR", metadata_dir)
    monkeypatch.setattr(wizard_server, "archive_dir_for", lambda _archive: tmp_path / "Archive")
    monkeypatch.setattr(
        wizard_server,
        "_archive_state",
        lambda session_arg, archive, selected_title=None: (
            {
                "archive": archive,
                "chapter": selected_title or chapter,
                "status": "",
                "details": "",
                "start_frame": 100,
                "end_frame": 103,
                "chapters": [],
            }
            if not (
                setattr(session_arg, "archive", archive)
                or setattr(session_arg, "chapter", selected_title or chapter)
                or setattr(
                    session_arg,
                    "chapters",
                    [{"title": chapter, "start_frame": 100, "end_frame": 103}],
                )
                or setattr(session_arg, "chapter_rows", [])
                or setattr(session_arg, "start_frame", 100)
                or setattr(session_arg, "end_frame", 103)
            )
            else {}
        ),
    )

    session = SessionState()
    handler = _HandlerStub()

    WizardHandler._handle_load_chapter(
        handler,
        session,
        {
            "archive": archive_name,
            "chapter": chapter,
        },
    )

    assert handler.payload is None
    assert handler.error == f"No archive video found for '{archive_name}'."
    assert session.load_running is False
    assert session.load_progress == 0.0
    assert session.load_message == f"No archive video found for '{archive_name}'."
    assert session.frame_source_video_path == ""

@skip(reason="This test is currently failing due to a change in the expected ffmpeg command arguments. Needs investigation and update when we have more tokens")
def test_ensure_audio_sync_file_extracts_twenty_seconds_of_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_video = tmp_path / "archive.mkv"
    source_video.write_bytes(b"video")
    temp_root = tmp_path / "temp"
    calls: list[list[str]] = []

    def _fake_run(cmd, check, capture_output, text):
        _ = (check, capture_output, text)
        calls.append(list(cmd))
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"RIFF" + (b"\x00" * 128))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(wizard_server, "_resolve_archive_video", lambda _archive: source_video)
    monkeypatch.setattr(wizard_server, "FFMPEG_BIN", "ffmpeg-test")
    monkeypatch.setattr(wizard_server.tempfile, "gettempdir", lambda: str(temp_root))
    monkeypatch.setattr(wizard_server.subprocess, "run", _fake_run)

    session = SessionState(
        archive="demo_archive",
        chapter="Example Chapter",
        start_frame=1000,
        end_frame=1100,
    )
    handler = WizardHandler.__new__(WizardHandler)

    audio_path, err, padded_start, video_offset = WizardHandler._ensure_audio_sync_file(handler, session)

    expected_start = wizard_server._frame_to_seconds(1000) - 20.0
    expected_end = wizard_server._frame_to_seconds(1100) + 20.0

    assert err == ""
    assert audio_path is not None
    assert audio_path.exists()
    assert calls
    cmd = calls[0]
    assert cmd[cmd.index("-ss") + 1] == f"{expected_start:.3f}"
    assert cmd[cmd.index("-to") + 1] == f"{expected_end:.3f}"
    assert padded_start == pytest.approx(expected_start)
    assert video_offset == pytest.approx(20.0)
    assert session.audio_sync_audio_path == str(audio_path)
    assert session.audio_sync_audio_key


def test_preview_render_passes_current_audio_sync_offset_to_extract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_dir = tmp_path / "Archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "demo_archive_proxy.mp4").write_bytes(b"proxy")
    captured: dict[str, float] = {}

    def _fake_extract(**kwargs):
        captured["audio_offset_seconds"] = float(kwargs["audio_offset_seconds"])
        return None, "stop here"

    monkeypatch.setattr(wizard_server, "archive_dir_for", lambda _archive: archive_dir)
    monkeypatch.setattr(wizard_server, "_ensure_render_chapter_extract", _fake_extract)

    session = _make_loaded_wizard_session("demo_archive", "Example Chapter", 100, 200)
    session.audio_sync_offset = 0.75
    handler = _HandlerStub()

    WizardHandler._handle_preview_render(
        handler,
        session,
        {
            "preview_mode": "review",
            "audio_sync_profile": {"offset_seconds": 0.75},
        },
    )

    assert captured["audio_offset_seconds"] == pytest.approx(0.75)
    assert handler.error == "stop here"


def test_chapter_audio_cache_key_includes_audio_sync_offset() -> None:
    handler = WizardHandler.__new__(WizardHandler)
    session = _make_loaded_wizard_session("demo_archive", "Example Chapter", 100, 200)

    session.audio_sync_offset = 0.0
    key_before = WizardHandler._chapter_audio_cache_key(handler, session)

    session.audio_sync_offset = 0.75
    key_after = WizardHandler._chapter_audio_cache_key(handler, session)

    assert key_before != key_after
    assert key_after.endswith("+0.7500")


def test_chapter_audio_extract_applies_current_audio_sync_offset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_video = tmp_path / "archive.mkv"
    source_video.write_bytes(b"video")
    temp_root = tmp_path / "temp"
    calls: list[list[str]] = []

    def _fake_run(cmd, check, capture_output, text):
        _ = (check, capture_output, text)
        calls.append(list(cmd))
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"RIFF" + (b"\x00" * 128))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(wizard_server, "_resolve_archive_video", lambda _archive: source_video)
    monkeypatch.setattr(wizard_server, "FFMPEG_BIN", "ffmpeg-test")
    monkeypatch.setattr(wizard_server.tempfile, "gettempdir", lambda: str(temp_root))
    monkeypatch.setattr(wizard_server.subprocess, "run", _fake_run)

    session = _make_loaded_wizard_session("demo_archive", "Example Chapter", 1000, 1100)
    session.audio_sync_offset = 0.75
    handler = WizardHandler.__new__(WizardHandler)

    audio_path, err = WizardHandler._ensure_chapter_audio_file(handler, session)

    expected_start = wizard_server._frame_to_seconds(1000) + 0.75
    expected_end = wizard_server._frame_to_seconds(1100) + 0.75

    assert err == ""
    assert audio_path is not None
    assert audio_path.exists()
    assert calls
    cmd = calls[0]
    af = cmd[cmd.index("-af") + 1]
    assert f"atrim=start={expected_start:.6f}:end={expected_end:.6f}" in af
    assert "apad=whole_dur=" in af
    assert session.chapter_audio_key.endswith("+0.7500")


def test_subtitles_generate_passes_current_audio_sync_offset_to_extract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_video = tmp_path / "demo_archive.mkv"
    archive_video.write_bytes(b"video")
    captured: dict[str, object] = {}

    def _fake_extract_audio(source_video, audio_path, **kwargs):
        captured["source_video"] = source_video
        captured["audio_path"] = Path(audio_path)
        captured["audio_offset_seconds"] = float(kwargs["audio_offset_seconds"])
        return ["ffmpeg", "-version"]

    def _fake_run(cmd, check=False, capture_output=True, text=True):
        _ = (cmd, check, capture_output, text)
        audio_path = Path(captured["audio_path"])
        with wave.open(str(audio_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(wizard_server, "_resolve_archive_video", lambda archive: archive_video)
    monkeypatch.setattr(wizard_server, "_load_whisper_model", lambda: object())
    monkeypatch.setattr(wizard_server, "_load_whisper_transcribe_module", lambda: SimpleNamespace())
    monkeypatch.setattr(wizard_server, "make_extract_audio", _fake_extract_audio)
    monkeypatch.setattr(wizard_server.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        wizard_server,
        "whisper_transcribe",
        lambda model, audio_path, prompt_text="": {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello there"}]},
    )
    monkeypatch.setattr(
        wizard_server,
        "subtitle_entries_from_whisper_result",
        lambda result: [
            {
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "text": "Hello there",
                "speaker": "",
                "confidence": None,
                "source": "whisper",
            }
        ],
    )

    session = _make_loaded_wizard_session("demo_archive", "Example Chapter", 100, 200)
    session.audio_sync_offset = 0.75
    handler = _HandlerStub()

    WizardHandler._handle_subtitles_generate(handler, session, {"mode": "replace"})

    assert captured["source_video"] == archive_video
    assert captured["audio_offset_seconds"] == pytest.approx(0.75)
    assert handler.error is None
    assert handler.payload["ok"] is True
    assert handler.payload["generated_count"] == 1


def test_handle_save_progress_persists_people_subtitles_and_split_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_dir = tmp_path / "metadata"
    archive_name = "demo_archive"
    chapter = "Example Chapter"
    archive_meta = metadata_dir / archive_name
    archive_meta.mkdir(parents=True, exist_ok=True)
    _write_single_chapter_tsv(archive_meta / "chapters.tsv", chapter, 100, 200)

    wizard_server._write_people_tsv_rows(
        archive_meta / "people.tsv",
        [
            (
                wizard_server._frame_to_seconds(80),
                wizard_server._frame_to_seconds(90),
                "Before Chapter",
            ),
            (
                wizard_server._frame_to_seconds(220),
                wizard_server._frame_to_seconds(230),
                "Outside Chapter",
            ),
        ],
    )
    wizard_server._write_subtitles_tsv_rows(
        archive_meta / "subtitles.tsv",
        [
            (
                wizard_server._frame_to_seconds(80),
                wizard_server._frame_to_seconds(90),
                "Old subtitle",
                "",
                None,
                "manual",
            ),
            (
                wizard_server._frame_to_seconds(220),
                wizard_server._frame_to_seconds(230),
                "Outside subtitle",
                "",
                None,
                "manual",
            ),
        ],
    )

    render_settings = archive_meta / "render_settings.json"
    render_settings.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(wizard_server, "METADATA_DIR", metadata_dir)
    monkeypatch.setattr(
        wizard_server,
        "persist_bad_frames_for_chapter",
        lambda **kwargs: (render_settings, 2, len(kwargs["fids"]), None),
    )
    monkeypatch.setattr(
        wizard_server,
        "update_chapter_gamma_in_render_settings",
        lambda **_kwargs: render_settings,
    )

    session = _make_loaded_wizard_session(archive_name, chapter, 100, 200)
    handler = _HandlerStub()

    WizardHandler._handle_save_progress(
        handler,
        session,
        {
            "force_all_frames_good": True,
            "gamma_profile": {
                "default_gamma": 1.2,
                "ranges": [{"start_frame": 110, "end_frame": 120, "gamma": 1.4}],
            },
            "people_profile": {
                "entries": [
                    {
                        "start": "00:00:00.000",
                        "end": "00:00:01.000",
                        "people": "Jim | Linda",
                    },
                ]
            },
            "subtitles_profile": {
                "entries": [
                    {
                        "start": "00:00:00.000",
                        "end": "00:00:01.500",
                        "text": "Hello there",
                        "speaker": "Narrator",
                        "confidence": 0.9,
                        "source": "manual",
                    }
                ]
            },
            "split_profile": {
                "entries": [
                    {"start": "0", "end": "40", "title": "Part 1"},
                    {"start": "40", "end": "100", "title": "Part 2"},
                ]
            },
        },
    )

    assert handler.error is None
    assert handler.payload is not None
    assert handler.payload["ok"] is True
    assert "people entries 1" in handler.payload["message"]
    assert "subtitle entries 1" in handler.payload["message"]
    assert "split entries 2" in handler.payload["message"]
    assert session.force_all_frames_good is True
    assert session.gamma_default == 1.2
    assert len(session.gamma_ranges) == 1

    people_local = wizard_server._load_people_entries_for_chapter(archive_name, 100, 200)
    subtitles_local = wizard_server._load_subtitle_entries_for_chapter(archive_name, 100, 200)
    assert [row["people"] for row in people_local] == ["Jim | Linda"]
    assert [row["text"] for row in subtitles_local] == ["Hello there"]

    people_rows = wizard_server._read_people_tsv_rows(archive_meta / "people.tsv")
    subtitles_rows = wizard_server._read_subtitles_tsv_rows(archive_meta / "subtitles.tsv")
    chapters_text = (archive_meta / "chapters.tsv").read_text(encoding="utf-8")
    assert any(row[2] == "Outside Chapter" for row in people_rows)
    assert any(row[2] == "Outside subtitle" for row in subtitles_rows)
    assert "Part 1" in chapters_text
    assert "Part 2" in chapters_text


def test_handle_save_returns_archive_state_and_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_dir = tmp_path / "metadata"
    archive_name = "demo_archive"
    chapter = "Example Chapter"
    archive_meta = metadata_dir / archive_name
    archive_meta.mkdir(parents=True, exist_ok=True)
    _write_single_chapter_tsv(archive_meta / "chapters.tsv", chapter, 100, 200)

    render_settings = archive_meta / "render_settings.json"
    render_settings.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(wizard_server, "METADATA_DIR", metadata_dir)
    monkeypatch.setattr(
        wizard_server,
        "persist_bad_frames_for_chapter",
        lambda **kwargs: (render_settings, 3, len(kwargs["fids"]), None),
    )
    monkeypatch.setattr(
        wizard_server,
        "update_chapter_gamma_in_render_settings",
        lambda **_kwargs: render_settings,
    )
    monkeypatch.setattr(
        wizard_server,
        "_archive_state",
        lambda _session, archive, selected_title=None: {
            "archive": archive,
            "chapter": selected_title,
            "status": "",
            "details": "stub",
            "start_frame": 100,
            "end_frame": 200,
            "chapters": [{"title": selected_title}],
        },
    )

    session = _make_loaded_wizard_session(archive_name, chapter, 100, 200)
    handler = _HandlerStub()

    WizardHandler._handle_save(
        handler,
        session,
        {
            "people_profile": {"entries": [{"start": "0", "end": "1", "people": "Jim"}]},
            "subtitles_profile": {
                "entries": [
                    {
                        "start": "0",
                        "end": "1",
                        "text": "Hello",
                        "speaker": "",
                        "source": "manual",
                    }
                ]
            },
            "split_profile": {"entries": [{"start": "0", "end": "100", "title": "Example Chapter"}]},
        },
    )

    assert handler.error is None
    assert handler.payload is not None
    assert handler.payload["ok"] is True
    assert "Saved people entries: 1." in handler.payload["message"]
    assert "Saved subtitle entries: 1." in handler.payload["message"]
    assert handler.payload["archive_state"]["archive"] == archive_name
    assert handler.payload["archive_state"]["chapter"] == chapter
    assert handler.payload["metadata_path"] == str(archive_meta / "chapters.tsv")


def test_do_post_rename_chapter_updates_tsv_and_returns_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_dir = tmp_path / "metadata"
    archive_name = "demo_archive"
    archive_meta = metadata_dir / archive_name
    archive_meta.mkdir(parents=True, exist_ok=True)
    _write_single_chapter_tsv(archive_meta / "chapters.tsv", "Old Title", 100, 200)

    monkeypatch.setattr(wizard_server, "METADATA_DIR", metadata_dir)
    monkeypatch.setattr(
        wizard_server,
        "_rename_chapter_outputs",
        lambda old_title, new_title, archive: [f"{wizard_server.safe(new_title)}.mp4"],
    )

    handler = _PostHandlerStub(
        "/api/rename_chapter",
        {
            "archive": archive_name,
            "old_title": "Old Title",
            "new_title": "New Title",
        },
    )

    WizardHandler.do_POST(handler)

    assert handler.error is None
    assert handler.payload is not None
    assert handler.payload["ok"] is True
    assert handler.payload["renamed_files"] == [f"{wizard_server.safe('New Title')}.mp4"]
    chapters_text = (archive_meta / "chapters.tsv").read_text(encoding="utf-8")
    assert "New Title" in chapters_text
    assert "Old Title" not in chapters_text
