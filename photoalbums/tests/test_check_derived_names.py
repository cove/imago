from argparse import Namespace

import numpy as np
import pytest

from photoalbums.scripts import check_derived_names as derived


@pytest.fixture
def placement(tmp_path, monkeypatch):
    archive = tmp_path / "Family_1980_B01_Archive"
    colorized = tmp_path / "colorized"
    archive.mkdir()
    colorized.mkdir()
    (archive / "Family_1980_B01_P02_D04-01.tif").write_bytes(b"archive")
    monkeypatch.setattr(derived, "_load_gray_thumb", lambda path: np.zeros((8, 8), dtype=np.uint8))
    monkeypatch.setattr(derived, "_orb_descriptors", lambda image: ([], None))
    monkeypatch.setattr(derived, "_orb_similarity", lambda *args: 1.0)
    return Namespace(
        archive_dir=archive, colorized_dir=colorized, threshold=0.1,
        execute=False, no_preview=True, width=60,
    )


@pytest.mark.parametrize("execute", [False, True])
def test_placements_allocate_distinct_iterations_without_overwriting(placement, execute, capsys):
    placement.execute = execute
    first = placement.colorized_dir / "a.png"
    second = placement.colorized_dir / "b.png"
    first.write_bytes(b"first colorization")
    second.write_bytes(b"second colorization")

    derived.cmd_place(placement)

    first_output = placement.archive_dir / "Family_1980_B01_P02_D04-02.png"
    second_output = placement.archive_dir / "Family_1980_B01_P02_D04-03.png"
    output = capsys.readouterr().out
    assert first_output.name in output
    assert second_output.name in output
    if execute:
        assert first_output.read_bytes() == first.read_bytes()
        assert second_output.read_bytes() == second.read_bytes()
    else:
        assert not first_output.exists()
        assert not second_output.exists()
    assert (placement.archive_dir / "Family_1980_B01_P02_D04-01.tif").read_bytes() == b"archive"


@pytest.mark.parametrize("execute", [False, True])
def test_existing_content_is_not_copied_again(placement, execute):
    placement.execute = execute
    source = placement.colorized_dir / "duplicate.png"
    source.write_bytes(b"archive")

    derived.cmd_place(placement)

    assert source.exists() is not execute
    renamed = placement.colorized_dir / "already copied - duplicate.png"
    assert renamed.exists() is execute
    assert len(list(placement.archive_dir.iterdir())) == 1


def test_zero_similarity_never_places_an_unmatched_image(placement, monkeypatch, capsys):
    placement.execute = True
    placement.threshold = 0.0
    (placement.colorized_dir / "unknown.png").write_bytes(b"unmatched")
    monkeypatch.setattr(derived, "_orb_similarity", lambda *args: 0.0)

    derived.cmd_place(placement)

    assert "NO MATCH" in capsys.readouterr().out
    assert len(list(placement.archive_dir.iterdir())) == 1
