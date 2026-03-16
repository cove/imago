"""
Playwright tests for the two flipbook-opening regressions:

Fix 1 — Resize suppression
  While the flipbook is in focus mode (fullscreen), a window `resize` event (e.g.
  from entering fullscreen) must NOT trigger scheduleVisibleRangeRefresh / renderFrameGridWindow,
  because doing so invalidates all the previously-loaded contact-sheet thumbnails with new
  viewport-dependent URLs, blanking out the frame grid.
  When focus mode is off, resize should trigger the refresh as normal.

Fix 2 — Secondary sheet fallback
  When the preferred no-metrics contact sheet for a frame hasn't loaded yet,
  `renderSparkPlaybackFrame` must draw immediately from any already-loaded sheet
  (e.g. the viewport-sized grid sheet) rather than making an async individual-URL request.
  `_findAnyLoadedSheetForFrameIndex` must parse start/count/columns from any sheet URL and
  return correct col/row sprite coordinates.
"""

import pytest

pytest.importorskip("playwright")

# ── shared fake state ─────────────────────────────────────────────────────────
# Six frames, fids 10-15, chapter starting at 10. Same layout as the freeze-sim tests.
# columns=6 metrics-based sheet URL: start=0, count=30, columns=6
#   → fid 10 index 0: col=0%6=0, row=0  sx=  0 sy=0
#   → fid 12 index 2: col=2%6=2, row=0  sx=320 sy=0
_METRICS_SHEET_URL = "/api/frame_contact_sheet?start=0&count=30&columns=6"
_METRICS_SX_INDEX_0 = 0  # col 0  (fid 10 / source)
_METRICS_SX_INDEX_2 = 320  # col 2  (fid 12 / bad)
_METRICS_SY = 0

_BASE_INJECT_JS = """
() => {
    state.chapters = [{ title: 'test', start_frame: 10, end_frame: 15 }];
    state.chapter  = 'test';

    state.review = {
        frames: [
            { fid: 10, status: 'good', image: '' },
            { fid: 11, status: 'good', image: '' },
            { fid: 12, status: 'bad',  image: '' },
            { fid: 13, status: 'bad',  image: '' },
            { fid: 14, status: 'good', image: '' },
            { fid: 15, status: 'good', image: '' },
        ],
        threshold: 0.5,
    };

    const frames = state.review.frames;
    sparkPlayFrames = frames;
    sparkPlayIndex  = 0;

    state.freezeReplacementMap = new Map([['12', 10]]);
    state.frameImages.set('10', 'data:image/jpeg;base64,SOURCEFRAME');
    state.frameImages.set('12', 'data:image/jpeg;base64,BADFRAME');
    state.simulateFreezeFrame = false;

    // Clear all loaded sheets so tests start from a known state.
    frameSheetImageObjects.clear();
}
"""


def _make_fake_sheet_js(url):
    """JS snippet that injects a completed fake Image into frameSheetImageObjects."""
    return f"""
() => {{
    const fakeSheet = new Image();
    Object.defineProperty(fakeSheet, 'complete',     {{ get: () => true }});
    Object.defineProperty(fakeSheet, 'naturalWidth', {{ get: () => 960  }});
    Object.defineProperty(fakeSheet, 'naturalHeight',{{ get: () => 120  }});
    frameSheetImageObjects.set('{url}', fakeSheet);
}}
"""


@pytest.fixture
def loading_page(page, live_server):
    page.goto(live_server)
    page.wait_for_load_state("networkidle")
    page.evaluate(_BASE_INJECT_JS)
    return page


# ── Fix 1: resize suppression ─────────────────────────────────────────────────


def test_flipbook_focus_mode_active_when_class_present(loading_page):
    """isFlipbookFocusModeActive() must reflect the flipbook-focus class on page2El."""
    result = loading_page.evaluate("""() => ({
        before: isFlipbookFocusModeActive(),
        afterAdd: (() => {
            if (page2El) page2El.classList.add('flipbook-focus');
            return isFlipbookFocusModeActive();
        })(),
        afterRemove: (() => {
            if (page2El) page2El.classList.remove('flipbook-focus');
            return isFlipbookFocusModeActive();
        })(),
    })""")
    assert result["before"] is False
    assert result["afterAdd"] is True
    assert result["afterRemove"] is False


def test_resize_skips_grid_render_in_flipbook_focus_mode(loading_page):
    """resize event must NOT call renderFrameGridWindow while flipbook-focus is active."""
    # Spy on renderFrameGridWindow and activate focus mode.
    loading_page.evaluate("""() => {
        if (page2El) page2El.classList.add('flipbook-focus');
        window._gridRenderCount = 0;
        const orig = window.renderFrameGridWindow;
        window.renderFrameGridWindow = function(...a) {
            window._gridRenderCount++;
            return orig && orig.apply(this, a);
        };
    }""")
    loading_page.evaluate("() => window.dispatchEvent(new Event('resize'))")
    loading_page.wait_for_timeout(
        120
    )  # let the rAF in scheduleVisibleRangeRefresh fire
    count = loading_page.evaluate("() => window._gridRenderCount || 0")
    assert (
        count == 0
    ), f"renderFrameGridWindow must not be called during flipbook focus mode, called {count}x"


def test_resize_triggers_grid_render_without_flipbook_focus(loading_page):
    """resize event must still call renderFrameGridWindow when flipbook-focus is absent."""
    loading_page.evaluate("""() => {
        if (page2El) page2El.classList.remove('flipbook-focus');
        window._gridRenderCount = 0;
        const orig = window.renderFrameGridWindow;
        window.renderFrameGridWindow = function(...a) {
            window._gridRenderCount++;
            return orig && orig.apply(this, a);
        };
    }""")
    loading_page.evaluate("() => window.dispatchEvent(new Event('resize'))")
    loading_page.wait_for_timeout(120)
    count = loading_page.evaluate("() => window._gridRenderCount || 0")
    assert (
        count > 0
    ), "renderFrameGridWindow must be called on resize when flipbook is not in focus mode"


# ── Fix 2: _findAnyLoadedSheetForFrameIndex ───────────────────────────────────


def test_find_any_sheet_returns_correct_sprite_for_covered_frame(loading_page):
    """_findAnyLoadedSheetForFrameIndex must return col/row for a frame covered by a loaded sheet."""
    loading_page.evaluate(_make_fake_sheet_js(_METRICS_SHEET_URL))
    result = loading_page.evaluate("""() => {
        const s0 = _findAnyLoadedSheetForFrameIndex(0);  // col=0%6=0, row=0
        const s2 = _findAnyLoadedSheetForFrameIndex(2);  // col=2%6=2, row=0
        const s7 = _findAnyLoadedSheetForFrameIndex(7);  // col=1, row=1
        return { s0, s2, s7 };
    }""")
    assert result["s0"] is not None
    assert result["s0"]["col"] == 0 and result["s0"]["row"] == 0

    assert result["s2"] is not None
    assert result["s2"]["col"] == 2 and result["s2"]["row"] == 0

    assert result["s7"] is not None
    assert result["s7"]["col"] == 1 and result["s7"]["row"] == 1  # 7%6=1, 7//6=1


def test_find_any_sheet_returns_null_for_uncovered_frame(loading_page):
    """_findAnyLoadedSheetForFrameIndex must return null when no loaded sheet covers the frame."""
    loading_page.evaluate(
        _make_fake_sheet_js(_METRICS_SHEET_URL)
    )  # covers indices 0-29
    result = loading_page.evaluate("() => _findAnyLoadedSheetForFrameIndex(30)")
    assert (
        result is None
    ), f"Index 30 is outside start=0 count=30, expected null, got {result!r}"


def test_find_any_sheet_returns_null_when_no_sheets_loaded(loading_page):
    """_findAnyLoadedSheetForFrameIndex must return null when frameSheetImageObjects is empty."""
    result = loading_page.evaluate("() => _findAnyLoadedSheetForFrameIndex(0)")
    assert result is None


# ── Fix 2: secondary sheet used in renderSparkPlaybackFrame ──────────────────


def _capture_drawimage_with_secondary_sheet(page, frame_index, sim_on):
    """
    Render a flipbook frame when only a metrics-based sheet is loaded (no no-metrics sheet).
    The secondary-sheet fallback in renderSparkPlaybackFrame should fire and drawImage with
    the correct coordinates derived from the metrics-based sheet.
    """
    return page.evaluate(f"""() => {{
        state.simulateFreezeFrame = {str(sim_on).lower()};

        let captured = null;
        const origDraw = CanvasRenderingContext2D.prototype.drawImage;
        CanvasRenderingContext2D.prototype.drawImage = function(img, sx, sy, sw, sh, ...rest) {{
            captured = {{ sx, sy, sw, sh }};
        }};

        renderSparkPlaybackFrame({frame_index});

        CanvasRenderingContext2D.prototype.drawImage = origDraw;
        return captured;
    }}""")


def test_secondary_sheet_draws_source_coords_when_sim_on(loading_page):
    """
    When only a metrics-based sheet is loaded and sim is ON, the secondary-sheet path
    must draw from the SOURCE frame's position (index 0, col 0 in 6-column sheet).
    """
    loading_page.evaluate(_make_fake_sheet_js(_METRICS_SHEET_URL))
    draw = _capture_drawimage_with_secondary_sheet(loading_page, 2, sim_on=True)
    assert (
        draw is not None
    ), "drawImage was not called — secondary sheet path did not fire"
    assert (
        draw["sx"] == _METRICS_SX_INDEX_0
    ), f"Expected sx={_METRICS_SX_INDEX_0} (source frame col 0), got {draw['sx']}"
    assert draw["sy"] == _METRICS_SY


def test_secondary_sheet_draws_bad_frame_coords_when_sim_off(loading_page):
    """
    When only a metrics-based sheet is loaded and sim is OFF, the secondary-sheet path
    must draw from the BAD frame's own position (index 2, col 2 in 6-column sheet).
    """
    loading_page.evaluate(_make_fake_sheet_js(_METRICS_SHEET_URL))
    draw = _capture_drawimage_with_secondary_sheet(loading_page, 2, sim_on=False)
    assert (
        draw is not None
    ), "drawImage was not called — secondary sheet path did not fire"
    assert (
        draw["sx"] == _METRICS_SX_INDEX_2
    ), f"Expected sx={_METRICS_SX_INDEX_2} (bad frame col 2), got {draw['sx']}"
    assert draw["sy"] == _METRICS_SY


def test_url_fallback_used_when_no_sheets_loaded(loading_page):
    """
    When no contact sheet is loaded at all, renderSparkPlaybackFrame must use the async
    URL-fallback path (Image.src is set) rather than drawing synchronously.
    """
    captured = loading_page.evaluate("""() => {
        let capturedSrc = null;
        const OrigImage = window.Image;
        window.Image = function() {
            const img = new OrigImage();
            Object.defineProperty(img, 'src', {
                set(v) { capturedSrc = v; },
                get() { return capturedSrc; },
                configurable: true,
            });
            return img;
        };
        window.Image.prototype = OrigImage.prototype;

        let drawCalled = false;
        const origDraw = CanvasRenderingContext2D.prototype.drawImage;
        CanvasRenderingContext2D.prototype.drawImage = function(...a) {
            drawCalled = true;
            return origDraw.apply(this, a);
        };

        renderSparkPlaybackFrame(0);  // fid 10, good frame

        CanvasRenderingContext2D.prototype.drawImage = origDraw;
        window.Image = OrigImage;
        return { src: capturedSrc, drawCalled };
    }""")
    # No sheets loaded → secondary path also misses → URL fallback fires
    assert captured["src"] is not None, "URL fallback should have set Image.src"
    assert (
        captured["drawCalled"] is False
    ), "drawImage should NOT be called synchronously when using the URL-fallback path"
