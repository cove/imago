"""
Playwright tests for freeze-frame simulation in the flipbook.

The `simulateFreezeFrame` checkbox should make the flipbook display the clean
source frame (and label it FROZEN) when a bad frame's cursor position is reached,
rather than showing the bad frame's own image (labelled BAD).

Covered scenarios — meta label:
  1. Sim OFF  + bad frame with replacement  → meta shows "BAD"
  2. Sim ON   + bad frame with replacement  → meta shows "FROZEN"
  3. Sim ON   + good frame                  → meta shows "GOOD"
  4. Sim ON   + bad frame with NO replacement → meta shows "BAD" (no clean source)
  5. Toggle sim off after it was on         → meta reverts to "BAD"

Covered scenarios — canvas image (fallback / URL path):
  6. Sim ON  → fallback loader uses source frame's image data
  7. Sim OFF → fallback loader uses bad frame's own image data

Covered scenarios — canvas image (sprite / contact-sheet path):
  8. Sim ON  → drawImage called with source frame's sprite coordinates (sx = col*W of source)
  9. Sim OFF → drawImage called with bad frame's own sprite coordinates (sx = col*W of bad)
"""

import pytest

pytest.importorskip("playwright")

# ── fake state ────────────────────────────────────────────────────────────────
# Frames: fids 10..15.  Frame 12 is bad with source 10.
#         Frame 13 is bad with no clean source.
#         All others are good.

_INJECT_JS = """
() => {
    // Minimal chapter span so _flipbookFrameGlobalIndex() can resolve.
    state.chapters = [{ title: 'test', start_frame: 10, end_frame: 15 }];
    state.chapter  = 'test';

    // Six frames: two bad, four good.
    const frames = [
        { fid: 10, status: 'good', image: '' },
        { fid: 11, status: 'good', image: '' },
        { fid: 12, status: 'bad',  image: '' },
        { fid: 13, status: 'bad',  image: '' },
        { fid: 14, status: 'good', image: '' },
        { fid: 15, status: 'good', image: '' },
    ];

    // Load frames into sparkPlayFrames directly.
    sparkPlayFrames = frames;
    sparkPlayIndex  = 0;

    // freezeReplacementMap: frame 12 → source 10; frame 13 has no entry.
    state.freezeReplacementMap = new Map([['12', 10]]);

    // Pre-load fake image data into frameImages so the fallback path returns
    // deterministic srcs without hitting the network.
    state.frameImages.set('10', 'data:image/jpeg;base64,SOURCEFRAME');
    state.frameImages.set('12', 'data:image/jpeg;base64,BADFRAME');

    // Ensure simulateFreezeFrame starts off.
    state.simulateFreezeFrame = false;
}
"""


@pytest.fixture
def freeze_page(page, live_server):
    """Page with minimal flipbook state injected."""
    page.goto(live_server)
    page.wait_for_load_state("networkidle")
    page.evaluate(_INJECT_JS)
    return page


# ── helpers ───────────────────────────────────────────────────────────────────


def _render_and_read_meta(page, frame_index):
    """Render a specific flipbook frame and return flipbookMetaEl.textContent."""
    return page.evaluate(f"""() => {{
        renderSparkPlaybackFrame({frame_index});
        return flipbookMetaEl ? flipbookMetaEl.textContent : '';
    }}""")


def _render_and_capture_image_src(page, frame_index):
    """
    Render a flipbook frame and return the image src that the canvas path would use.

    Forces the sprite path to miss (frameSheetImageObjects is empty in test context
    because state.review.frames is not set, so _flipbookFrameGlobalIndex returns -1),
    then captures the src set on the fallback Image element via a spy.
    """
    return page.evaluate(f"""() => {{
        // Spy on Image to capture which src the fallback loader receives.
        let capturedSrc = null;
        const OrigImage = window.Image;
        window.Image = function() {{
            const img = new OrigImage();
            Object.defineProperty(img, 'src', {{
                set(v) {{ capturedSrc = v; }},
                get() {{ return capturedSrc; }},
                configurable: true,
            }});
            return img;
        }};
        window.Image.prototype = OrigImage.prototype;

        renderSparkPlaybackFrame({frame_index});

        window.Image = OrigImage;
        return capturedSrc;
    }}""")


# ── tests ─────────────────────────────────────────────────────────────────────


def test_sim_off_bad_frame_shows_BAD(freeze_page):
    """With sim disabled, a bad frame must be labelled BAD."""
    freeze_page.evaluate("() => { state.simulateFreezeFrame = false; }")
    meta = _render_and_read_meta(freeze_page, 2)  # fid 12, bad, has replacement
    assert "BAD" in meta, f"Expected BAD in meta, got: {meta!r}"
    assert "FROZEN" not in meta


def test_sim_on_bad_frame_with_replacement_shows_FROZEN(freeze_page):
    """With sim enabled, a bad frame that has a clean source must be labelled FROZEN."""
    freeze_page.evaluate("() => { state.simulateFreezeFrame = true; }")
    meta = _render_and_read_meta(freeze_page, 2)  # fid 12, bad, source = fid 10
    assert "FROZEN" in meta, f"Expected FROZEN in meta, got: {meta!r}"
    assert "BAD" not in meta


def test_sim_on_good_frame_shows_GOOD(freeze_page):
    """With sim enabled, a good frame must still be labelled GOOD."""
    freeze_page.evaluate("() => { state.simulateFreezeFrame = true; }")
    meta = _render_and_read_meta(freeze_page, 0)  # fid 10, good
    assert "GOOD" in meta, f"Expected GOOD in meta, got: {meta!r}"


def test_sim_on_bad_frame_without_replacement_shows_BAD(freeze_page):
    """A bad frame with no entry in freezeReplacementMap must still show BAD."""
    freeze_page.evaluate("() => { state.simulateFreezeFrame = true; }")
    meta = _render_and_read_meta(freeze_page, 3)  # fid 13, bad, no replacement
    assert "BAD" in meta, f"Expected BAD in meta (no source), got: {meta!r}"
    assert "FROZEN" not in meta


def test_toggle_sim_off_reverts_frozen_to_BAD(freeze_page):
    """Disabling sim after it was on must revert FROZEN back to BAD."""
    freeze_page.evaluate("() => { state.simulateFreezeFrame = true; }")
    meta_on = _render_and_read_meta(freeze_page, 2)
    assert "FROZEN" in meta_on, f"Setup: expected FROZEN, got: {meta_on!r}"

    freeze_page.evaluate("() => { state.simulateFreezeFrame = false; }")
    meta_off = _render_and_read_meta(freeze_page, 2)
    assert "BAD" in meta_off, f"Expected BAD after toggling sim off, got: {meta_off!r}"
    assert "FROZEN" not in meta_off


# ── sprite (contact-sheet) path setup ────────────────────────────────────────
#
# With FRAME_SHEET_DEFAULT_COLUMNS=8 and 6 review frames starting at fid 10:
#   count = max(8, min(512, 6)) = 8   (padded to fill one column-row)
#   sheet URL = /api/frame_contact_sheet?start=0&count=8&columns=8
#   thumbWidth=160, thumbHeight=120
#
#   fid 10 → index 0  → col=0, row=0 → sx=   0, sy=0
#   fid 12 → index 2  → col=2, row=0 → sx= 320, sy=0

_SPRITE_SX_SOURCE = 0  # sx for fid 10 (source frame, index 0)
_SPRITE_SX_BAD = 320  # sx for fid 12 (bad   frame, index 2)
_SPRITE_SY = 0  # both frames are in row 0

_SETUP_SPRITE_STATE_JS = """
() => {
    // Populate state.review.frames so _flipbookFrameGlobalIndex() returns real indices.
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

    // Build the expected contact-sheet URL exactly as frameContactSheetSpecForSheetStart
    // would, then inject a complete (already-loaded) fake Image so the sprite path fires.
    const sheetUrl = '/api/frame_contact_sheet?start=0&count=8&columns=8';
    const fakeSheet = new Image();
    Object.defineProperty(fakeSheet, 'complete',      { get: () => true });
    Object.defineProperty(fakeSheet, 'naturalWidth',  { get: () => 1280 });
    Object.defineProperty(fakeSheet, 'naturalHeight', { get: () => 120  });
    frameSheetImageObjects.set(sheetUrl, fakeSheet);
}
"""


def _render_and_capture_drawimage(page, frame_index):
    """
    Render a flipbook frame via the sprite (contact-sheet) path and return the
    drawImage source-rect args {sx, sy, sw, sh} captured from the canvas context.

    Requires _SETUP_SPRITE_STATE_JS to have been called first.
    """
    return page.evaluate(f"""() => {{
        let captured = null;
        const origDraw = CanvasRenderingContext2D.prototype.drawImage;
        CanvasRenderingContext2D.prototype.drawImage = function(img, sx, sy, sw, sh, ...rest) {{
            // Only capture the 9-arg form used by the sprite path.
            captured = {{ sx, sy, sw, sh }};
            // Don't actually draw — img is a fake with no pixel data.
        }};

        renderSparkPlaybackFrame({frame_index});

        CanvasRenderingContext2D.prototype.drawImage = origDraw;
        return captured;
    }}""")


# ── canvas image source tests ─────────────────────────────────────────────────


def test_sim_on_canvas_uses_source_frame_image(freeze_page):
    """With sim on, the fallback image loader must use the SOURCE frame's image data."""
    freeze_page.evaluate("() => { state.simulateFreezeFrame = true; }")
    src = _render_and_capture_image_src(freeze_page, 2)  # fid 12 (bad), source = fid 10
    assert src == "data:image/jpeg;base64,SOURCEFRAME", f"Canvas should load source frame image, got: {src!r}"


def test_sim_off_canvas_uses_bad_frame_image(freeze_page):
    """With sim off, the fallback image loader must use the BAD frame's own image data."""
    freeze_page.evaluate("() => { state.simulateFreezeFrame = false; }")
    src = _render_and_capture_image_src(freeze_page, 2)  # fid 12 (bad)
    assert src == "data:image/jpeg;base64,BADFRAME", (
        f"Canvas should load bad frame's own image when sim is off, got: {src!r}"
    )


# ── canvas sprite (contact-sheet) path tests ──────────────────────────────────


def test_sim_on_sprite_uses_source_frame_coordinates(freeze_page):
    """
    With sim on and the contact sheet loaded, drawImage must use the SOURCE
    frame's sprite coordinates (sx for index 0), not the bad frame's (index 2).
    """
    freeze_page.evaluate(_SETUP_SPRITE_STATE_JS)
    freeze_page.evaluate("() => { state.simulateFreezeFrame = true; }")
    draw = _render_and_capture_drawimage(freeze_page, 2)  # fid 12 (bad), source = fid 10
    assert draw is not None, "drawImage was not called — sprite path did not fire"
    assert draw["sx"] == _SPRITE_SX_SOURCE, f"Expected sx={_SPRITE_SX_SOURCE} (source frame col 0), got sx={draw['sx']}"
    assert draw["sy"] == _SPRITE_SY


def test_sim_off_sprite_uses_bad_frame_coordinates(freeze_page):
    """
    With sim off, drawImage must use the BAD frame's own sprite coordinates
    (sx for index 2), not the source frame's (index 0).
    """
    freeze_page.evaluate(_SETUP_SPRITE_STATE_JS)
    freeze_page.evaluate("() => { state.simulateFreezeFrame = false; }")
    draw = _render_and_capture_drawimage(freeze_page, 2)  # fid 12 (bad)
    assert draw is not None, "drawImage was not called — sprite path did not fire"
    assert draw["sx"] == _SPRITE_SX_BAD, f"Expected sx={_SPRITE_SX_BAD} (bad frame col 2), got sx={draw['sx']}"
    assert draw["sy"] == _SPRITE_SY
