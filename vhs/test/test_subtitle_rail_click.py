"""
Test that clicking a subtitle in the flipbook subtitle rail seeks the video to that subtitle.

The subtitle rail renders rows with `data-subtitle-start-seconds` embedded on each row.
A click handler on the rail calls renderSparkPlaybackFrame with the frame index corresponding
to the clicked subtitle's start_seconds.
"""

import pytest

pytest.importorskip("playwright")

_BASE_INJECT_JS = """
() => {
    state.chapters = [{ title: 'test', start_frame: 0, end_frame: 150 }];
    state.chapter  = 'test';

    // 6 frames at 5 fps → frames cover 0..1.0s
    state.review = {
        frames: [
            { fid: 0,  status: 'good', image: '' },
            { fid: 5,  status: 'good', image: '' },
            { fid: 10, status: 'good', image: '' },
            { fid: 15, status: 'good', image: '' },
            { fid: 20, status: 'good', image: '' },
            { fid: 25, status: 'good', image: '' },
        ],
        threshold: 0.5,
    };

    sparkPlayFrames = state.review.frames;
    sparkPlayIndex  = 0;

    state.subtitlesProfile = {
        entries: [
            { start_seconds: 0.0, end_seconds: 1.0, start: '00:00:00.000', end: '00:00:01.000', text: 'Hello world', speaker: '', confidence: null, source: '' },
            { start_seconds: 2.0, end_seconds: 3.0, start: '00:00:02.000', end: '00:00:03.000', text: 'Second line', speaker: '', confidence: null, source: '' },
            { start_seconds: 4.0, end_seconds: 5.0, start: '00:00:04.000', end: '00:00:05.000', text: 'Third line',  speaker: '', confidence: null, source: '' },
        ],
        source: 'test',
    };

    // Activate subtitles step (step 5 = subtitles mode)
    state.wizardStep = 5;

    // Enable flipbook focus + subtitle-mode on page2El
    if (page2El) {
        page2El.classList.add('flipbook-focus');
        page2El.classList.add('flipbook-subtitle-mode');
    }
    if (flipbookPreviewEl) {
        flipbookPreviewEl.classList.add('active');
    }
}
"""


@pytest.fixture
def subtitle_rail_page(page, live_server):
    page.goto(live_server)
    page.wait_for_load_state("networkidle")
    page.evaluate(_BASE_INJECT_JS)
    # Render the rail directly with a pre-built `around` so no range-filter applies.
    page.evaluate("""() => {
        const entries = [
            { start_seconds: 0.0, end_seconds: 1.0, start: '00:00:00.000', end: '00:00:01.000', text: 'Hello world', speaker: '', confidence: null, source: '' },
            { start_seconds: 2.0, end_seconds: 3.0, start: '00:00:02.000', end: '00:00:03.000', text: 'Second line', speaker: '', confidence: null, source: '' },
            { start_seconds: 4.0, end_seconds: 5.0, start: '00:00:04.000', end: '00:00:05.000', text: 'Third line',  speaker: '', confidence: null, source: '' },
        ];
        const around = { entries, active: [entries[0]], prev: null, next: null };
        renderFlipbookSubtitleRail(0, around);
    }""")
    return page


def test_subtitle_rail_rows_have_start_seconds_attribute(subtitle_rail_page):
    """Each rendered subtitle rail row must carry data-subtitle-start-seconds."""
    rows = subtitle_rail_page.evaluate("""() => {
        if (!flipbookSubtitleRailEl) return [];
        return Array.from(
            flipbookSubtitleRailEl.querySelectorAll('[data-subtitle-start-seconds]')
        ).map(el => parseFloat(el.getAttribute('data-subtitle-start-seconds')));
    }""")
    assert rows == [
        0.0,
        2.0,
        4.0,
    ], f"Expected start_seconds [0.0, 2.0, 4.0] on rail rows, got {rows}"


def test_clicking_subtitle_rail_row_calls_render_with_correct_seconds(
    subtitle_rail_page,
):
    """Clicking a subtitle rail row must call renderSparkPlaybackFrame at the right time."""
    result = subtitle_rail_page.evaluate("""() => {
        let capturedIdx = null;
        const orig = window.renderSparkPlaybackFrame;
        window.renderSparkPlaybackFrame = function(idx, opts) {
            capturedIdx = idx;
            // Do not actually render — avoid canvas/sheet dependencies.
        };

        // Click the row for start_seconds=2.0 (second subtitle)
        const row = flipbookSubtitleRailEl &&
            flipbookSubtitleRailEl.querySelector('[data-subtitle-start-seconds="2"]');
        if (row) row.click();

        window.renderSparkPlaybackFrame = orig;
        return { called: capturedIdx !== null, idx: capturedIdx };
    }""")
    assert result["called"], "renderSparkPlaybackFrame was not called after clicking a subtitle row"


def test_clicking_subtitle_rail_row_uses_start_seconds_for_seek(subtitle_rail_page):
    """The frame index passed to renderSparkPlaybackFrame must match flipbookIndexFromChapterSeconds(start_seconds)."""
    result = subtitle_rail_page.evaluate("""() => {
        let capturedIdx = null;
        const orig = window.renderSparkPlaybackFrame;
        window.renderSparkPlaybackFrame = function(idx, opts) {
            capturedIdx = idx;
        };

        const startSec = 2.0;
        const expectedIdx = flipbookIndexFromChapterSeconds(startSec);

        const row = flipbookSubtitleRailEl &&
            flipbookSubtitleRailEl.querySelector('[data-subtitle-start-seconds="2"]');
        if (row) row.click();

        window.renderSparkPlaybackFrame = orig;
        return { capturedIdx, expectedIdx };
    }""")
    assert result["capturedIdx"] == result["expectedIdx"], (
        f"Expected frame index {result['expectedIdx']} (from flipbookIndexFromChapterSeconds(2.0)), "
        f"got {result['capturedIdx']}"
    )


def test_clicking_child_element_in_row_also_seeks(subtitle_rail_page):
    """Clicking the text div inside a row (not the row itself) must still seek correctly."""
    result = subtitle_rail_page.evaluate("""() => {
        let capturedIdx = null;
        const orig = window.renderSparkPlaybackFrame;
        window.renderSparkPlaybackFrame = function(idx, opts) {
            capturedIdx = idx;
        };

        // Click the inner text div of the first subtitle row
        const textEl = flipbookSubtitleRailEl &&
            flipbookSubtitleRailEl.querySelector('[data-subtitle-start-seconds="0"] .flipbook-subtitle-rail-text');
        if (textEl) textEl.click();

        window.renderSparkPlaybackFrame = orig;
        return { called: capturedIdx !== null };
    }""")
    assert result["called"], (
        "renderSparkPlaybackFrame was not called when clicking a child element inside a subtitle row"
    )
