import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["TEST_ENV"] = "1"
from common import *

TESTDATA_DIR = BASE / "test" / "test_data"
TEST_ARCHIVE_FIXTURE = TESTDATA_DIR / "test_01_archive.mkv"
os.environ["PYTHONPATH"] = str(BASE)

ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def test_step_drive_checksums():
    print("Testing drive checksum generation and verification...")
    from vhs_pipeline.checksum import generate_drive_checksum, verify_drive

    assert generate_drive_checksum() == 0
    assert verify_drive() == 0
    print("Test step_drive_checksums: PASSED.")
    DRIVE_CHECKSUM_FILE.unlink()


def test_sha3_generate_and_verify():
    print("Testing SHA3-256 generate + verify...")
    test_root = ARCHIVE_DIR / "_sha3_test"
    test_root.mkdir(parents=True, exist_ok=True)
    test_file = test_root / "hello.txt"
    test_file.write_text("hello sha3", encoding="utf-8")

    manifest = test_root / "sha3-manifest.txt"
    write_sha3_manifest(test_root, manifest, relative_base=test_root)
    rc = verify_manifest(test_root, manifest, algo="sha3")
    assert rc == 0

    manifest.unlink()
    test_file.unlink()
    test_root.rmdir()
    print("Test SHA3-256 generate + verify: PASSED.")


def test_blake3_verify_only():
    print("Testing BLAKE3 verify (legacy)...")
    test_root = ARCHIVE_DIR / "_blake3_test"
    test_root.mkdir(parents=True, exist_ok=True)
    test_file = test_root / "hello.txt"
    test_file.write_text("hello blake3", encoding="utf-8")

    manifest = test_root / "blake3-manifest.txt"
    old_cwd = os.getcwd()
    os.chdir(test_root)
    try:
        r = subprocess.run([str(B3SUM_BIN), test_file.name], capture_output=True, text=True)
        assert r.returncode == 0
        manifest.write_text(r.stdout, encoding="utf-8")
    finally:
        os.chdir(old_cwd)

    rc = verify_manifest(test_root, manifest, algo="blake3")
    assert rc == 0

    manifest.unlink()
    test_file.unlink()
    test_root.rmdir()
    print("Test BLAKE3 verify (legacy): PASSED.")


def test_vhs_tuner_toggle_override_cycle():
    print("Testing vhs_tuner frame-toggle override cycle...")
    import vhs_tuner

    fids = [100, 101, 102]
    sigs = {
        "chroma": np.array([0.0, 10.0, 20.0], dtype=np.float64),
        "noise": np.zeros(3, dtype=np.float64),
        "tear": np.zeros(3, dtype=np.float64),
        "wave": np.zeros(3, dtype=np.float64),
    }

    # 3-state manual override:
    # auto-good -> force bad -> clear
    # auto-bad  -> force good -> clear
    overrides = vhs_tuner.toggle_frame_override(
        fid=100,
        fids=fids,
        sigs=sigs,
        overrides={},
        wc=1.0,
        wn=0.0,
        wt=0.0,
        ww=0.0,
        tm="value",
        ik=3.5,
        tv=0.0,
        bp=10.0,
    )
    assert overrides.get(100) == "bad"
    overrides = vhs_tuner.toggle_frame_override(
        fid=100,
        fids=fids,
        sigs=sigs,
        overrides=overrides,
        wc=1.0,
        wn=0.0,
        wt=0.0,
        ww=0.0,
        tm="value",
        ik=3.5,
        tv=0.0,
        bp=10.0,
    )
    assert 100 not in overrides

    overrides = vhs_tuner.toggle_frame_override(
        fid=102,
        fids=fids,
        sigs=sigs,
        overrides={},
        wc=1.0,
        wn=0.0,
        wt=0.0,
        ww=0.0,
        tm="value",
        ik=3.5,
        tv=0.0,
        bp=10.0,
    )
    assert overrides.get(102) == "good"
    overrides = vhs_tuner.toggle_frame_override(
        fid=102,
        fids=fids,
        sigs=sigs,
        overrides=overrides,
        wc=1.0,
        wn=0.0,
        wt=0.0,
        ww=0.0,
        tm="value",
        ik=3.5,
        tv=0.0,
        bp=10.0,
    )
    assert 102 not in overrides

    # Regression check: signal sparkline HTML should include the red cut line.
    scores = vhs_tuner.combined_score(sigs, 1.0, 0.0, 0.0, 0.0)
    thr = vhs_tuner.compute_threshold(scores, "value", 3.5, 0.0, 10.0)
    sc_ch, sc_no, sc_te, sc_wa, _ = vhs_tuner.build_sparklines_html(
        sigs=sigs,
        scores=scores,
        threshold=thr,
        weights=(0.2, 0.3, 0.4, 0.5),  # type: ignore[arg-type]
    )
    for svg in (sc_ch, sc_no, sc_te, sc_wa):
        assert 'stroke="#e03030"' in svg

    print("Test vhs_tuner frame-toggle override cycle: PASSED.")


def _write_unit_chapters_ffmetadata(meta_root: Path, bad_csv: str = "") -> Path:
    cf = meta_root / "unit_archive" / "chapters.ffmetadata"
    cf.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        ";FFMETADATA1",
        "[CHAPTER]",
        "TIMEBASE=1001/30000",
        "START=1000",
        "END=1100",
        "TITLE=Unit Chapter",
        f"BAD_FRAMES={bad_csv}",
        "",
    ]
    cf.write_text("\n".join(lines), encoding="utf-8")
    return cf


def test_vhs_tuner_manual_click_persists_bad_frames():
    print("Testing vhs_tuner manual click persistence to chapters BAD_FRAMES...")
    import tempfile
    import time

    import vhs_tuner

    fids = [1000]
    sigs = {
        "chroma": np.array([0.0], dtype=np.float64),
        "noise": np.array([0.0], dtype=np.float64),
        "tear": np.array([0.0], dtype=np.float64),
        "wave": np.array([0.0], dtype=np.float64),
    }

    old_meta = vhs_tuner.METADATA_DIR
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vhs_tuner.METADATA_DIR = root
        try:
            cf = _write_unit_chapters_ffmetadata(root, bad_csv="")
            overrides, last_click, dbg = vhs_tuner.apply_manual_click_override(
                raw_click="1000:1000",
                fids=fids,
                sigs=sigs,
                overrides={},
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=1.0,
                bp=10.0,
                last_click_event={"fid": -1, "ts": -1},
            )
            assert overrides.get(1000) == "bad", dbg
            chapters = vhs_tuner.load_archive_chapters(cf)
            ch = vhs_tuner._find_chapter(chapters, "Unit Chapter")
            assert ch is not None
            assert ch.get("bad_frames", []) == []

            _p, _n = vhs_tuner._persist_visible_bad_frames(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=fids,
                sigs=sigs,
                overrides=overrides,
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=1.0,
                bp=10.0,
            )
            chapters = vhs_tuner.load_archive_chapters(cf)
            ch = vhs_tuner._find_chapter(chapters, "Unit Chapter")
            assert ch is not None
            assert ch.get("bad_frames", []) == [1000]

            time.sleep(0.30)
            overrides2, _last2, dbg2 = vhs_tuner.apply_manual_click_override(
                raw_click="1000:1400",
                fids=fids,
                sigs=sigs,
                overrides=overrides,
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=1.0,
                bp=10.0,
                last_click_event=last_click,
            )
            assert 1000 not in overrides2, dbg2
            _p, _n = vhs_tuner._persist_visible_bad_frames(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=fids,
                sigs=sigs,
                overrides=overrides2,
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=1.0,
                bp=10.0,
            )
            chapters = vhs_tuner.load_archive_chapters(cf)
            ch = vhs_tuner._find_chapter(chapters, "Unit Chapter")
            assert ch is not None
            assert ch.get("bad_frames", []) == []
        finally:
            vhs_tuner.METADATA_DIR = old_meta
    print("Test vhs_tuner manual click persistence to chapters BAD_FRAMES: PASSED.")


def test_vhs_tuner_click_dedupe_prevents_double_toggle():
    print("Testing vhs_tuner click dedupe for duplicate events...")
    import tempfile

    import vhs_tuner

    fids = [1000]
    sigs = {
        "chroma": np.array([0.0], dtype=np.float64),
        "noise": np.array([0.0], dtype=np.float64),
        "tear": np.array([0.0], dtype=np.float64),
        "wave": np.array([0.0], dtype=np.float64),
    }

    old_meta = vhs_tuner.METADATA_DIR
    with tempfile.TemporaryDirectory() as td:
        vhs_tuner.METADATA_DIR = Path(td)
        try:
            overrides, last_click, dbg = vhs_tuner.apply_manual_click_override(
                raw_click="1000:2000",
                fids=fids,
                sigs=sigs,
                overrides={},
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=1.0,
                bp=10.0,
                last_click_event={"fid": -1, "ts": -1},
            )
            assert overrides.get(1000) == "bad", dbg

            overrides2, last2, dbg2 = vhs_tuner.apply_manual_click_override(
                raw_click="1000:2050",
                fids=fids,
                sigs=sigs,
                overrides=overrides,
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=1.0,
                bp=10.0,
                last_click_event=last_click,
            )
            assert overrides2 == overrides
            assert last2 == last_click
            assert "ignored: duplicate click" in dbg2
        finally:
            vhs_tuner.METADATA_DIR = old_meta
    print("Test vhs_tuner click dedupe for duplicate events: PASSED.")


def test_vhs_tuner_manual_click_modes_bad_and_good():
    print("Testing vhs_tuner manual click mark modes (bad/good/clear)...")
    import tempfile
    import time

    import vhs_tuner

    fids = [1000]
    sigs = {
        "chroma": np.array([0.0], dtype=np.float64),
        "noise": np.array([0.0], dtype=np.float64),
        "tear": np.array([0.0], dtype=np.float64),
        "wave": np.array([0.0], dtype=np.float64),
    }

    old_meta = vhs_tuner.METADATA_DIR
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vhs_tuner.METADATA_DIR = root
        try:
            cf = _write_unit_chapters_ffmetadata(root, bad_csv="")
            ov_bad, last_bad, dbg_bad = vhs_tuner.apply_manual_click_override(
                raw_click="1000:1000",
                fids=fids,
                sigs=sigs,
                overrides={},
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=1.0,
                bp=10.0,
                mark_mode="bad",
                last_click_event={"fid": -1, "ts": -1},
            )
            assert ov_bad.get(1000) == "bad", dbg_bad
            _p, _n = vhs_tuner._persist_visible_bad_frames(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=fids,
                sigs=sigs,
                overrides=ov_bad,
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=1.0,
                bp=10.0,
            )
            ch = vhs_tuner._find_chapter(vhs_tuner.load_archive_chapters(cf), "Unit Chapter")
            assert ch is not None
            assert ch.get("bad_frames", []) == [1000]
            text = cf.read_text(encoding="utf-8")
            assert "BAD_FRAME_OVERRIDE=" not in text
            assert "GOOD_FRAME_OVERRIDE=" not in text

            time.sleep(0.30)
            ov_good, last_good, dbg_good = vhs_tuner.apply_manual_click_override(
                raw_click="1000:1300",
                fids=fids,
                sigs=sigs,
                overrides=ov_bad,
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=1.0,
                bp=10.0,
                mark_mode="good",
                last_click_event=last_bad,
            )
            assert ov_good.get(1000) == "good", dbg_good
            _p, _n = vhs_tuner._persist_visible_bad_frames(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=fids,
                sigs=sigs,
                overrides=ov_good,
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=1.0,
                bp=10.0,
            )
            ch = vhs_tuner._find_chapter(vhs_tuner.load_archive_chapters(cf), "Unit Chapter")
            assert ch is not None
            assert ch.get("bad_frames", []) == []
            text = cf.read_text(encoding="utf-8")
            assert "BAD_FRAME_OVERRIDE=" not in text
            assert "GOOD_FRAME_OVERRIDE=" not in text

            time.sleep(0.30)
            ov_clear, _last_clear, dbg_clear = vhs_tuner.apply_manual_click_override(
                raw_click="1000:1600",
                fids=fids,
                sigs=sigs,
                overrides=ov_good,
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=1.0,
                bp=10.0,
                mark_mode="clear",
                last_click_event=last_good,
            )
            assert 1000 not in ov_clear, dbg_clear
            _p, _n = vhs_tuner._persist_visible_bad_frames(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=fids,
                sigs=sigs,
                overrides=ov_clear,
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=1.0,
                bp=10.0,
            )
            text = cf.read_text(encoding="utf-8")
            assert "BAD_FRAME_OVERRIDE=" not in text
            assert "GOOD_FRAME_OVERRIDE=" not in text
        finally:
            vhs_tuner.METADATA_DIR = old_meta
    print("Test vhs_tuner manual click mark modes (bad/good/clear): PASSED.")


def test_vhs_tuner_auto_and_manual_persist_to_bad_frames():
    print("Testing vhs_tuner auto + manual persistence to chapters BAD_FRAMES...")
    import tempfile

    import vhs_tuner

    old_meta = vhs_tuner.METADATA_DIR
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vhs_tuner.METADATA_DIR = root
        try:
            cf = _write_unit_chapters_ffmetadata(root, bad_csv="1005,1007")
            fids = [1000, 1001, 1002]
            sigs = {
                "chroma": np.array([0.0, 10.0, 0.0], dtype=np.float64),
                "noise": np.zeros(3, dtype=np.float64),
                "tear": np.zeros(3, dtype=np.float64),
                "wave": np.zeros(3, dtype=np.float64),
            }
            # Manual bad at frame 1000; auto should mark frame 1001 bad at threshold 0.
            _path, _count = vhs_tuner._persist_visible_bad_frames(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=fids,
                sigs=sigs,
                overrides={1000: "bad"},
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=0.0,
                bp=10.0,
            )
            chapters = vhs_tuner.load_archive_chapters(cf)
            ch = vhs_tuner._find_chapter(chapters, "Unit Chapter")
            assert ch is not None
            out = set(int(x) for x in ch.get("bad_frames", []))
            # Existing unsampled values preserved + manual + auto (global IDs).
            assert {1000, 1001, 1005, 1007}.issubset(out), (
                f"persisted BAD_FRAMES missing expected values: {sorted(out)}"
            )
        finally:
            vhs_tuner.METADATA_DIR = old_meta

    print("Test vhs_tuner auto + manual persistence to chapters BAD_FRAMES: PASSED.")


def test_vhs_tuner_persist_loaded_frame_set_mode():
    print("Testing vhs_tuner BAD_FRAMES persistence from loaded frame set only...")
    import tempfile

    import vhs_tuner

    old_meta = vhs_tuner.METADATA_DIR
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vhs_tuner.METADATA_DIR = root
        try:
            cf = _write_unit_chapters_ffmetadata(root, bad_csv="")

            path, count, analyzed, err = vhs_tuner.persist_bad_frames_for_chapter(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=[1000, 1001],
                sigs={
                    "chroma": np.array([0.0, 10.0], dtype=np.float64),
                    "noise": np.array([0.0, 0.0], dtype=np.float64),
                    "tear": np.array([0.0, 0.0], dtype=np.float64),
                    "wave": np.array([0.0, 0.0], dtype=np.float64),
                },
                overrides={1000: "bad"},
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=0.0,
                bp=10.0,
                progress=None,
            )
            assert not err, err
            assert path == cf
            assert analyzed == 2
            assert count == 2

            chapters = vhs_tuner.load_archive_chapters(cf)
            ch = vhs_tuner._find_chapter(chapters, "Unit Chapter")
            assert ch is not None
            assert ch.get("bad_frames", []) == [1000, 1001]
        finally:
            vhs_tuner.METADATA_DIR = old_meta

    print("Test vhs_tuner BAD_FRAMES persistence from loaded frame set only: PASSED.")


def test_vhs_tuner_chapter_bad_overrides_half_open_range():
    print("Testing vhs_tuner ignores persisted override metadata lines...")
    import tempfile

    import vhs_tuner

    old_meta = vhs_tuner.METADATA_DIR
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vhs_tuner.METADATA_DIR = root
        try:
            cf = root / "unit_archive" / "chapters.ffmetadata"
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(
                ";FFMETADATA1\n"
                "[CHAPTER]\n"
                "TIMEBASE=1001/30000\n"
                "START=1000\n"
                "END=1100\n"
                "TITLE=Unit Chapter\n"
                "BAD_FRAME_OVERRIDE=1099,1100\n",
                encoding="utf-8",
            )
            out = vhs_tuner._chapter_bad_overrides(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
            )
            assert out == {}
        finally:
            vhs_tuner.METADATA_DIR = old_meta

    print("Test vhs_tuner ignores persisted override metadata lines: PASSED.")


def test_update_chapter_bad_frames_preserves_untouched_chapters():
    print("Testing BAD_FRAMES updates preserve untouched chapter blocks...")
    import tempfile

    from common import parse_chapters, update_chapter_bad_frames_in_ffmetadata

    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "chapters.ffmetadata"
        cf.write_text(
            ";FFMETADATA1\n"
            "[CHAPTER]\n"
            "TIMEBASE=1001/30000\n"
            "START=0\n"
            "END=100\n"
            "TITLE=Chap A\n"
            "BAD_FRAMES=1,2\n"
            "[CHAPTER]\n"
            "TIMEBASE=1001/30000\n"
            "START=100\n"
            "END=200\n"
            "TITLE=Chap B\n"
            "BAD_FRAMES=3,4\n",
            encoding="utf-8",
        )
        touched = update_chapter_bad_frames_in_ffmetadata(cf, {"Chap A": [9, 10]})
        assert touched == 1

        _ffm, chapters = parse_chapters(cf)
        by_title = {str(ch.get("title", "")).strip(): str(ch.get("bad_frames", "")).strip() for ch in chapters}
        assert by_title.get("Chap A") == "9,10"
        assert by_title.get("Chap B") == "3,4"

    print("Test BAD_FRAMES update preserves untouched chapter blocks: PASSED.")


def test_update_chapter_bad_frames_omits_empty_line():
    print("Testing BAD_FRAMES empty updates remove BAD_FRAMES line...")
    import tempfile

    from common import update_chapter_bad_frames_in_ffmetadata

    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "chapters.ffmetadata"
        cf.write_text(
            ";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1001/30000\nSTART=0\nEND=100\nTITLE=Chap A\nBAD_FRAMES=1,2\n",
            encoding="utf-8",
        )
        touched = update_chapter_bad_frames_in_ffmetadata(cf, {"Chap A": []})
        assert touched == 1
        text = cf.read_text(encoding="utf-8")
        assert "BAD_FRAMES=" not in text

    print("Test BAD_FRAMES empty updates remove BAD_FRAMES line: PASSED.")


def test_vhs_tuner_plain_wizard_entrypoint():
    print("Testing vhs_tuner plain HTML wizard entrypoint...")
    src = (ROOT / "vhs_tuner.py").read_text(encoding="utf-8", errors="ignore")

    assert "from apps.plain_html_wizard.server import run as run_plain_wizard" in src
    assert 'run_plain_wizard(host="0.0.0.0", port=8092)' in src
    assert "from libs.vhs_tuner_core import *" in src

    print("Test vhs_tuner plain HTML wizard entrypoint: PASSED.")


def test_runtime_scripts_do_not_generate_framemd5():
    print("Testing runtime scripts do not generate framemd5/md5 temp outputs...")
    tuner_src = (ROOT / "vhs_tuner.py").read_text(encoding="utf-8", errors="ignore").lower()
    assert "framemd5" not in tuner_src
    assert ".md5" not in tuner_src
    print("Test runtime scripts do not generate framemd5/md5 temp outputs: PASSED.")


def main():
    print("Running tests...")
    test_step_drive_checksums()
    test_sha3_generate_and_verify()
    test_blake3_verify_only()
    test_vhs_tuner_toggle_override_cycle()
    test_vhs_tuner_manual_click_persists_bad_frames()
    test_vhs_tuner_click_dedupe_prevents_double_toggle()
    test_vhs_tuner_manual_click_modes_bad_and_good()
    test_vhs_tuner_auto_and_manual_persist_to_bad_frames()
    test_vhs_tuner_persist_loaded_frame_set_mode()
    test_vhs_tuner_chapter_bad_overrides_half_open_range()
    test_update_chapter_bad_frames_preserves_untouched_chapters()
    test_update_chapter_bad_frames_omits_empty_line()
    test_vhs_tuner_plain_wizard_entrypoint()
    test_runtime_scripts_do_not_generate_framemd5()


if __name__ == "__main__":
    main()
