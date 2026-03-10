"""
Playwright tests for subtitle editor <-> audio/frame timeline scroll synchronisation.

These tests start a real (headless) Chromium browser against a live WizardHandler
server, inject minimal fake state via page.evaluate(), and assert that:

  1. Scrubbing the timeline scrolls the subtitle editor to the active row.
  2. Scrolling the subtitle editor updates the timeline scrubber index.
  3. Programmatic scrolls (from timeline scrub) do NOT set the user-scrolling
     lock, so subsequent timeline scrubs continue to move the editor.
"""

import pytest

pytest.importorskip("playwright")

# ── test parameters ──────────────────────────────────────────────────────────

_FRAME_COUNT = 60   # number of fake frames  (fid 0 .. 59)
_SUB_COUNT   = 20   # subtitle rows — enough to overflow a 200 px-tall container
# Each subtitle entry covers 3 frames.
# Frame i  →  chapterLocalSecondsFromFid(i) = i * (1001/30000)
# Entry j  →  start = j*3 * FPS,  end = (j+1)*3 * FPS − 0.001


# ── helpers ──────────────────────────────────────────────────────────────────

_INJECT_STATE_JS = f"""
() => {{
    const FPS = 1001 / 30000;
    const framesPerEntry = {_FRAME_COUNT // _SUB_COUNT};

    // Minimal fake chapter so chapterLocalSecondsFromFid() works.
    state.chapters = [{{ title: 'test', start_frame: 0, end_frame: {_FRAME_COUNT - 1} }}];
    state.chapter  = 'test';

    // Fake review with {_FRAME_COUNT} frames (fid == index).
    state.review = {{
        frames: Array.from({{length: {_FRAME_COUNT}}}, (_, i) => ({{ fid: i }})),
        threshold: 0.5,
    }};

    // {_SUB_COUNT} subtitle entries, each spanning framesPerEntry frames.
    state.subtitlesProfile = {{
        entries: Array.from({{length: {_SUB_COUNT}}}, (_, j) => {{
            const start = j * framesPerEntry * FPS;
            const end   = (j + 1) * framesPerEntry * FPS - 0.001;
            return {{
                start_seconds: start,
                end_seconds:   end,
                start: formatTimestampSeconds(start),
                end:   formatTimestampSeconds(end),
                text:  'Line ' + (j + 1),
            }};
        }}),
        source: 'test',
    }};

    state.wizardStep = 5;  // 'subtitles' step

    // Sync the range-input with our fake frame count.
    if (timelineScrubEl) {{
        timelineScrubEl.min   = '0';
        timelineScrubEl.max   = '{_FRAME_COUNT - 1}';
        timelineScrubEl.value = '0';
    }}

    // Show page2 and the subtitle editor with a fixed height so it scrolls.
    const page2 = document.getElementById('page2');
    if (page2) page2.style.display = 'flex';
    if (subtitlesEditorEl) {{
        subtitlesEditorEl.classList.remove('hidden-ui');
        subtitlesEditorEl.style.height   = '200px';
        subtitlesEditorEl.style.overflow = 'auto';
    }}

    // Render subtitle rows and timeline.
    refreshSubtitlesEditorFromState();
    renderPeopleTimeline(0);
}}
"""


@pytest.fixture
def subtitle_page(page, live_server):
    """Browser page loaded with fake subtitle state injected."""
    page.goto(live_server)
    page.wait_for_load_state("networkidle")
    page.evaluate(_INJECT_STATE_JS)
    page.wait_for_selector("tr[data-sub-row='1']")
    return page


# ── tests ────────────────────────────────────────────────────────────────────

def test_scrub_timeline_scrolls_subtitle_editor(subtitle_page):
    """Scrubbing to the last frame should scroll the subtitle editor to the last row."""
    assert subtitle_page.evaluate("() => subtitlesEditorEl.scrollTop") == 0

    # Read scrollTop inside the same evaluate() call so the rAF-deferred
    # refreshVisibleRangeFromGrid() cannot fire and reset it before we check.
    scroll_after = subtitle_page.evaluate(f"""() => {{
        scrubTimelineToIndex({_FRAME_COUNT - 1});
        return subtitlesEditorEl.scrollTop;
    }}""")

    assert scroll_after > 0, (
        "Subtitle editor should scroll down when the timeline is scrubbed to the last frame"
    )


def test_subtitle_scroll_updates_timeline_index(subtitle_page):
    """Scrolling the subtitle editor to the last row should advance the scrubber."""
    idx_before = subtitle_page.evaluate("() => Number(timelineScrubEl.value)")
    assert idx_before == 0

    subtitle_page.evaluate("""() => {
        const rows = Array.from(subtitlesEditorEl.querySelectorAll('tr[data-sub-row="1"]'));
        const last = rows[rows.length - 1];
        if (last) subtitlesEditorEl.scrollTop = last.offsetTop;
        subtitlesEditorEl.dispatchEvent(new Event('scroll'));
    }""")
    subtitle_page.wait_for_timeout(150)

    idx_after = subtitle_page.evaluate("() => Number(timelineScrubEl.value)")
    assert idx_after > idx_before, (
        "Timeline scrubber should advance when the subtitle editor is scrolled to a later row"
    )


def test_programmatic_scroll_does_not_lock_future_syncs(subtitle_page):
    """
    A programmatic scroll triggered by scrubTimelineToIndex() must NOT set
    subtitleEditorUserScrolling=true, which would block the next sync.
    """
    last = _FRAME_COUNT - 1

    # Scrub to last frame and capture flag + scrollTop before rAF fires.
    result = subtitle_page.evaluate(f"""() => {{
        scrubTimelineToIndex({last});
        return {{
            scrollTop: subtitlesEditorEl.scrollTop,
            userScrolling: subtitleEditorUserScrolling,
        }};
    }}""")

    assert result["scrollTop"] > 0, "Editor should scroll to last row"
    assert result["userScrolling"] is False, (
        "subtitleEditorUserScrolling should be False after a programmatic scroll"
    )

    # Scrubbing back to 0 should move the editor back to the top.
    scroll_back = subtitle_page.evaluate("""() => {
        scrubTimelineToIndex(0);
        return subtitlesEditorEl.scrollTop;
    }""")
    assert scroll_back == 0, (
        "Subtitle editor should return to top when scrubbed back to frame 0"
    )
