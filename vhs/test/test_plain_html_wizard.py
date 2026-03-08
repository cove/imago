import pytest
pytest.importorskip("numpy")
pytest.importorskip("cv2")

from pathlib import Path
import shutil

import numpy as np

from common import update_chapter_bad_frames_in_render_settings
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
    html = INDEX_HTML.read_text(encoding="utf-8")

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
    assert 'id="subtitlesEditor"' in html
    assert 'id="peopleMeta"' in html
    assert 'id="subtitlesMeta"' in html
    assert "subtitles-editor-grid" in html
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
        update_chapter_bad_frames_in_render_settings(
            archive,
            {title: [100, 105, 205]},
        )
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
    assert lines[0].startswith("__ffmeta_header\t__ffmeta_order\t__chapter_order\t__chapter_index\tffmeta_title\tTIMEBASE\tSTART\tEND\ttitle")
    assert "parent_chapter" not in lines[0]
    assert "start_frame" not in lines[0]
    assert lines[1].split("\t")[5:9] == ["1001/30000", "100", "200", chapter]
    assert lines[2].split("\t")[5:9] == ["1001/30000", "100", "125", "Part 1"]
    assert lines[3].split("\t")[5:9] == ["1001/30000", "125", "180", "Part 2"]


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
                "__ffmeta_header\t__ffmeta_order\t__chapter_order\t__chapter_index\tffmeta_title\tTIMEBASE\tSTART\tEND\ttitle",
                f";FFMETADATA1\ttitle\tTIMEBASE|START|END|title\t1\tDemo Archive\t1001/30000\t100\t200\t{chapter}",
                ";FFMETADATA1\ttitle\tTIMEBASE|START|END|title\t2\tDemo Archive\t1001/30000\t100\t125\tPart 1",
                ";FFMETADATA1\ttitle\tTIMEBASE|START|END|title\t3\tDemo Archive\t1001/30000\t125\t180\tPart 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _load_split_entries_for_chapter(archive, chapter, 100, 200)

    assert rows == [
        {"start_frame": 0, "end_frame": 25, "start": "0", "end": "25", "title": "Part 1"},
        {"start_frame": 25, "end_frame": 80, "start": "25", "end": "80", "title": "Part 2"},
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
        {"start_frame": 0, "end_frame": 25, "start": "0", "end": "25", "title": "Part 1"},
        {"start_frame": 25, "end_frame": 80, "start": "25", "end": "80", "title": "Part 2"},
    ]
