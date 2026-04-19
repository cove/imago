from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from photoalbums.lib.caption_layout_migration import (
    _crop_region_index,
    find_sidecars_with_legacy_caption_layout,
    migrate_album_caption_layout,
    migrate_sidecar_caption_layout,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")
    return path


def test_find_sidecars_with_legacy_caption_layout_reports_page_and_crop_sidecars(tmp_path: Path) -> None:
    page_xmp = _write(
        tmp_path / "Egypt_1975_B00_Pages" / "Egypt_1975_B00_P26_V.xmp",
        """
        <?xml version="1.0" encoding="utf-8"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:imago="https://imago.local/ns/1.0/">
          <rdf:RDF>
            <rdf:Description rdf:about="">
              <dc:description>
                <rdf:Alt>
                  <rdf:li xml:lang="x-default">TEMPLE OF HEAVEN
        NO SMOKING</rdf:li>
                  <rdf:li xml:lang="x-scene">NO SMOKING</rdf:li>
                </rdf:Alt>
              </dc:description>
              <imago:OCRText>TEMPLE OF HEAVEN
        NO SMOKING</imago:OCRText>
              <imago:SceneText>NO SMOKING</imago:SceneText>
            </rdf:Description>
          </rdf:RDF>
        </x:xmpmeta>
        """,
    )
    crop_xmp = _write(
        tmp_path / "Egypt_1975_B00_Photos" / "Egypt_1975_B00_P26_D01-00_V.xmp",
        """
        <?xml version="1.0" encoding="utf-8"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:imago="https://imago.local/ns/1.0/">
          <rdf:RDF>
            <rdf:Description rdf:about="">
              <dc:description>
                <rdf:Alt>
                  <rdf:li xml:lang="x-default">TEMPLE OF HEAVEN
        NO SMOKING</rdf:li>
                  <rdf:li xml:lang="x-caption">Legacy crop caption</rdf:li>
                </rdf:Alt>
              </dc:description>
              <imago:OCRText>TEMPLE OF HEAVEN
        NO SMOKING</imago:OCRText>
            </rdf:Description>
          </rdf:RDF>
        </x:xmpmeta>
        """,
    )

    assert find_sidecars_with_legacy_caption_layout(tmp_path) == [page_xmp, crop_xmp]


def test_migrate_album_caption_layout_rewrites_page_and_crop_sidecars_in_place(tmp_path: Path) -> None:
    page_xmp = _write(
        tmp_path / "Egypt_1975_B00_Pages" / "Egypt_1975_B00_P26_V.xmp",
        """
        <?xml version="1.0" encoding="utf-8"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:imago="https://imago.local/ns/1.0/" xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/" xmlns:custom="https://example.com/custom/1.0/">
          <rdf:RDF>
            <rdf:Description rdf:about="">
              <custom:KeepMe>Preserve page field</custom:KeepMe>
              <dc:description>
                <rdf:Alt>
                  <rdf:li xml:lang="x-default">TEMPLE OF HEAVEN
        NO SMOKING</rdf:li>
                  <rdf:li xml:lang="x-caption">Ignored summary</rdf:li>
                  <rdf:li xml:lang="x-scene">NO SMOKING</rdf:li>
                </rdf:Alt>
              </dc:description>
              <imago:OCRText>TEMPLE OF HEAVEN
        NO SMOKING</imago:OCRText>
              <imago:SceneText>NO SMOKING</imago:SceneText>
              <mwg-rs:RegionInfo>
                <mwg-rs:RegionList>
                  <rdf:Bag>
                    <rdf:li rdf:parseType="Resource" mwg-rs:Name="Authoritative region caption" />
                  </rdf:Bag>
                </mwg-rs:RegionList>
              </mwg-rs:RegionInfo>
            </rdf:Description>
          </rdf:RDF>
        </x:xmpmeta>
        """,
    )
    crop_xmp = _write(
        tmp_path / "Egypt_1975_B00_Photos" / "Egypt_1975_B00_P26_D01-00_V.xmp",
        """
        <?xml version="1.0" encoding="utf-8"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:imago="https://imago.local/ns/1.0/" xmlns:custom="https://example.com/custom/1.0/">
          <rdf:RDF>
            <rdf:Description rdf:about="">
              <custom:KeepMe>Preserve crop field</custom:KeepMe>
              <dc:description>
                <rdf:Alt>
                  <rdf:li xml:lang="x-default">TEMPLE OF HEAVEN
        NO SMOKING</rdf:li>
                  <rdf:li xml:lang="x-caption">Legacy crop caption</rdf:li>
                </rdf:Alt>
              </dc:description>
              <imago:OCRText>TEMPLE OF HEAVEN
        NO SMOKING</imago:OCRText>
            </rdf:Description>
          </rdf:RDF>
        </x:xmpmeta>
        """,
    )

    result = migrate_album_caption_layout(tmp_path)

    assert result == {"files_scanned": 2, "files_changed": 2}

    page_xml = page_xmp.read_text(encoding="utf-8")
    assert "Preserve page field" in page_xml
    assert "OCR:\nTEMPLE OF HEAVEN\nNO SMOKING\n\nScene Text:\nNO SMOKING" in page_xml
    assert 'xml:lang="x-caption"' not in page_xml
    assert 'xml:lang="x-scene"' not in page_xml

    crop_xml = crop_xmp.read_text(encoding="utf-8")
    assert "Preserve crop field" in crop_xml
    assert 'xml:lang="x-default">Authoritative region caption</rdf:li>' in crop_xml
    assert "<imago:ParentOCRText>TEMPLE OF HEAVEN\nNO SMOKING</imago:ParentOCRText>" in crop_xml
    assert 'xml:lang="x-caption"' not in crop_xml
    assert "<imago:OCRText>TEMPLE OF HEAVEN\nNO SMOKING</imago:OCRText>" not in crop_xml
    assert find_sidecars_with_legacy_caption_layout(tmp_path) == []


def test_migrate_sidecar_caption_layout_leaves_current_sidecar_untouched(tmp_path: Path) -> None:
    crop_xmp = _write(
        tmp_path / "Egypt_1975_B00_Photos" / "Egypt_1975_B00_P26_D01-00_V.xmp",
        """
        <?xml version="1.0" encoding="utf-8"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:imago="https://imago.local/ns/1.0/" xmlns:custom="https://example.com/custom/1.0/">
          <rdf:RDF>
            <rdf:Description rdf:about="">
              <custom:KeepMe>Preserve crop field</custom:KeepMe>
              <dc:description>
                <rdf:Alt>
                  <rdf:li xml:lang="x-default">Authoritative region caption</rdf:li>
                </rdf:Alt>
              </dc:description>
              <imago:ParentOCRText>TEMPLE OF HEAVEN
        NO SMOKING</imago:ParentOCRText>
            </rdf:Description>
          </rdf:RDF>
        </x:xmpmeta>
        """,
    )
    original = crop_xmp.read_text(encoding="utf-8")

    changed = migrate_sidecar_caption_layout(crop_xmp)

    assert changed is False
    assert crop_xmp.read_text(encoding="utf-8") == original


def test_crop_region_index_accounts_for_archive_derived_offset(tmp_path: Path) -> None:
    archive_dir = tmp_path / "TheOrient_1974_B00_Archive"
    pages_dir = tmp_path / "TheOrient_1974_B00_Pages"
    photos_dir = tmp_path / "TheOrient_1974_B00_Photos"
    archive_dir.mkdir()
    pages_dir.mkdir()
    photos_dir.mkdir()

    (archive_dir / "TheOrient_1974_B00_P44_D01-01.png").write_bytes(b"derived")
    (archive_dir / "TheOrient_1974_B00_P44_D02-01.png").write_bytes(b"derived")

    crop_xmp = photos_dir / "TheOrient_1974_B00_P44_D03-00_V.xmp"

    assert _crop_region_index(crop_xmp) == 0
