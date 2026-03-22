"""Playwright test for subtitle editor row deletion."""

import pytest

pytest.importorskip("playwright")

_INJECT_STATE_JS = """
() => {
    state.chapters = [{ title: 'test', start_frame: 0, end_frame: 29 }];
    state.chapter = 'test';
    state.review = {
        frames: Array.from({ length: 30 }, (_, i) => ({ fid: i })),
        threshold: 0.5,
    };
    state.subtitlesProfile = {
        entries: [
            {
                start_seconds: 0,
                end_seconds: 1,
                start: formatTimestampSeconds(0),
                end: formatTimestampSeconds(1),
                text: 'Opening line',
            },
            {
                start_seconds: 1.1,
                end_seconds: 2.1,
                start: formatTimestampSeconds(1.1),
                end: formatTimestampSeconds(2.1),
                text: 'Second line',
            },
        ],
        source: 'test',
    };
    setStep(6);
    refreshSubtitlesEditorFromState();
}
"""


@pytest.fixture
def subtitle_delete_page(page, live_server):
    page.goto(live_server)
    page.wait_for_load_state("networkidle")
    page.evaluate(_INJECT_STATE_JS)
    page.wait_for_selector("button[data-subtitle-row-delete='0']")
    return page


def test_clicking_subtitle_delete_button_removes_row(subtitle_delete_page):
    before = subtitle_delete_page.evaluate("""() => ({
        rowCount: subtitlesEditorEl.querySelectorAll('button[data-subtitle-row-delete]').length,
        texts: Array.from(subtitlesEditorEl.querySelectorAll('input[data-sub-field="text"]')).map((el) => el.value),
    })""")
    assert before["rowCount"] == 2
    assert before["texts"][:2] == ["Opening line", "Second line"]

    subtitle_delete_page.click("button[data-subtitle-row-delete='0']")

    after = subtitle_delete_page.evaluate("""() => ({
        rowCount: subtitlesEditorEl.querySelectorAll('button[data-subtitle-row-delete]').length,
        texts: Array.from(subtitlesEditorEl.querySelectorAll('input[data-sub-field="text"]')).map((el) => el.value),
        profileTexts: (state.subtitlesProfile.entries || []).map((row) => row.text),
    })""")
    assert after["rowCount"] == 1
    assert after["texts"][:1] == ["Second line"]
    assert after["profileTexts"] == ["Second line"]
