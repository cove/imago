"""Face region reconciliation across page, scan, and crop XMP sources.

Resolves IPTC4xmpExt:ImageRegion face boxes from all sources belonging to a page
into a single canonical page-space coordinate system, then backfills missing boxes
back into each source.

Pipeline position: after face-refresh/immich-face-refresh, before any downstream
consumer that expects consistent face metadata across all source files.

Entry point: reconcile_page(page_xmp, *, dry_run=True)
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..naming import (
    DERIVED_NAME_RE,
    SCAN_NAME_RE,
    SCAN_TIFF_RE,
    archive_dir_for_album_dir,
    is_pages_dir,
    is_photos_dir,
    pages_dir_for_album_dir,
    parse_album_filename,
    photos_dir_for_album_dir,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XMP namespace constants (subset needed here)
# ---------------------------------------------------------------------------

_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_IPTC_EXT_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
_MWGRS_NS = "http://www.metadataworkinggroup.com/schemas/regions/"
_STAREA_NS = "http://ns.adobe.com/xap/1.0/sType/Area#"
_X_NS = "adobe:ns:meta/"

_RDF_BAG = f"{{{_RDF_NS}}}Bag"
_RDF_LI = f"{{{_RDF_NS}}}li"
_RDF_ALT = f"{{{_RDF_NS}}}Alt"
_RDF_DESC = f"{{{_RDF_NS}}}Description"
_RDF_ROOT = f"{{{_RDF_NS}}}RDF"
_RDF_PARSE_TYPE = f"{{{_RDF_NS}}}parseType"

_IPTC_IMAGE_REGION = f"{{{_IPTC_EXT_NS}}}ImageRegion"
_IPTC_REGION_BOUNDARY = f"{{{_IPTC_EXT_NS}}}RegionBoundary"
_IPTC_RCTYPE = f"{{{_IPTC_EXT_NS}}}RCtype"
_IPTC_RID = f"{{{_IPTC_EXT_NS}}}rId"
_IPTC_NAME = f"{{{_IPTC_EXT_NS}}}Name"
_IPTC_RB_SHAPE = f"{{{_IPTC_EXT_NS}}}rbShape"
_IPTC_RB_UNIT = f"{{{_IPTC_EXT_NS}}}rbUnit"
_IPTC_RB_X = f"{{{_IPTC_EXT_NS}}}rbX"
_IPTC_RB_Y = f"{{{_IPTC_EXT_NS}}}rbY"
_IPTC_RB_W = f"{{{_IPTC_EXT_NS}}}rbW"
_IPTC_RB_H = f"{{{_IPTC_EXT_NS}}}rbH"

_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

# Threshold for IoU-based clustering and duplicate detection
_DEFAULT_IOU_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SourceFaceBox:
    """A face bounding box read from a source XMP, in that source's coordinate space."""

    name: str
    rx: float  # top-left x, normalized 0–1
    ry: float  # top-left y, normalized 0–1
    rw: float  # width, normalized 0–1
    rh: float  # height, normalized 0–1
    source_kind: str  # "page" | "scan" | "crop"
    source_xmp: Path


@dataclass
class ProjectedFaceBox:
    """A face box from a source, projected into page coordinate space."""

    name: str
    page_rx: float
    page_ry: float
    page_rw: float
    page_rh: float
    confidence: str  # "high" | "approximate" | "unresolved"
    origin: SourceFaceBox


@dataclass
class FaceCluster:
    """A group of face boxes that overlap enough to be the same face on the page."""

    resolved_rx: float
    resolved_ry: float
    resolved_rw: float
    resolved_rh: float
    names: list[str]  # distinct non-empty names across all sources
    sources: list[ProjectedFaceBox]
    has_conflict: bool  # True when multiple distinct non-empty names exist


@dataclass
class PendingWrite:
    """A face region write that would be applied during backfill."""

    xmp_path: Path
    source_kind: str
    name: str
    rx: float  # coordinates in the target source's coordinate space
    ry: float
    rw: float
    rh: float
    reason: str  # human-readable explanation


@dataclass
class ReconcileResult:
    page_xmp: Path
    clusters: list[FaceCluster]
    conflicts: list[FaceCluster]
    unresolved_sources: list[SourceFaceBox]
    pending_writes: list[PendingWrite]
    sources_scanned: list[Path]


# ---------------------------------------------------------------------------
# IPTC face region reading
# ---------------------------------------------------------------------------


def _is_face_li(li: ET.Element) -> bool:
    rctype = str(li.findtext(_IPTC_RCTYPE, default="") or "").strip().lower()
    if rctype.startswith("face-"):
        return True
    rid = str(li.findtext(_IPTC_RID, default="") or "").strip()
    return rid.startswith("face-")


def _read_iptc_name(li: ET.Element) -> str:
    name_el = li.find(_IPTC_NAME)
    if name_el is None:
        return ""
    alt = name_el.find(_RDF_ALT)
    if alt is not None:
        for item in alt.findall(_RDF_LI):
            text = str(item.text or "").strip()
            if text:
                return text
    return str(name_el.text or "").strip()


def read_iptc_face_boxes(xmp_path: Path) -> list[dict[str, Any]]:
    """Parse IPTC4xmpExt:ImageRegion face boxes from an XMP sidecar.

    Returns a list of dicts with keys: name, rx, ry, rw, rh (normalized 0–1).
    Returns [] if the file is absent, malformed, or has no face regions.
    """
    if not xmp_path.is_file():
        return []
    try:
        tree = ET.parse(str(xmp_path))
    except ET.ParseError:
        return []

    results: list[dict[str, Any]] = []
    for li in tree.iter(_RDF_LI):
        if not _is_face_li(li):
            continue
        boundary = li.find(_IPTC_REGION_BOUNDARY)
        if boundary is None:
            continue
        unit = str(boundary.findtext(_IPTC_RB_UNIT, default="") or "").strip().lower()
        if unit not in {"relative", ""}:
            continue
        try:
            rx = float(boundary.findtext(_IPTC_RB_X, default="0") or 0)
            ry = float(boundary.findtext(_IPTC_RB_Y, default="0") or 0)
            rw = float(boundary.findtext(_IPTC_RB_W, default="0") or 0)
            rh = float(boundary.findtext(_IPTC_RB_H, default="0") or 0)
        except (TypeError, ValueError):
            continue
        if rw <= 0 or rh <= 0:
            continue
        results.append({
            "name": _read_iptc_name(li),
            "rx": rx,
            "ry": ry,
            "rw": rw,
            "rh": rh,
        })
    return results


# ---------------------------------------------------------------------------
# Page source discovery
# ---------------------------------------------------------------------------


def _page_prefix(page_xmp: Path) -> str:
    """Return the base page prefix, e.g. 'Egypt_1975_B00_P26' from 'Egypt_1975_B00_P26_V.xmp'."""
    stem = page_xmp.stem  # e.g. Egypt_1975_B00_P26_V
    if stem.endswith("_V"):
        return stem[:-2]
    return stem


def find_archive_scans_for_page(page_xmp: Path) -> list[Path]:
    """Return sorted list of archive scan XMPs for the same page as page_xmp."""
    if not is_pages_dir(page_xmp.parent):
        return []
    archive_dir = archive_dir_for_album_dir(page_xmp.parent)
    if not archive_dir.is_dir():
        return []
    _, _, _, page_str = parse_album_filename(page_xmp.name)
    if not page_str.isdigit():
        return []
    page_int = int(page_str)
    scans: list[Path] = sorted(
        p.with_suffix(".xmp")
        for p in archive_dir.iterdir()
        for sm in (SCAN_TIFF_RE.match(p.name),)
        if sm and int(sm.group("page")) == page_int and p.with_suffix(".xmp").is_file()
    )
    return scans


def find_crop_xmps_for_page(page_xmp: Path) -> list[Path]:
    """Return sorted list of crop XMPs (_D##-##_V.xmp) in the _Photos/ dir for this page."""
    if not is_pages_dir(page_xmp.parent):
        return []
    photos_dir = photos_dir_for_album_dir(page_xmp.parent)
    if not photos_dir.is_dir():
        return []
    prefix = _page_prefix(page_xmp)
    pattern = re.compile(rf"^{re.escape(prefix)}_D\d{{2}}-\d{{2}}_V\.xmp$", re.IGNORECASE)
    return sorted(p for p in photos_dir.iterdir() if pattern.match(p.name))


# ---------------------------------------------------------------------------
# Page photo regions (mwg-rs:RegionList) in normalized coords
# ---------------------------------------------------------------------------


def read_page_photo_regions(page_xmp: Path) -> list[dict[str, float]]:
    """Read mwg-rs:RegionList Photo regions from page XMP in normalized coords.

    Returns list of dicts with keys: index (0-based), cx, cy, nw, nh.
    cx, cy = center; nw, nh = normalized width/height (0–1).
    """
    if not page_xmp.is_file():
        return []
    try:
        tree = ET.parse(str(page_xmp))
    except ET.ParseError:
        return []

    results: list[dict[str, float]] = []
    idx = 0
    for li in tree.iter(_RDF_LI):
        rtype = li.get(f"{{{_MWGRS_NS}}}Type")
        if rtype != "Photo":
            continue
        cx_t = li.get(f"{{{_STAREA_NS}}}x")
        cy_t = li.get(f"{{{_STAREA_NS}}}y")
        nw_t = li.get(f"{{{_STAREA_NS}}}w")
        nh_t = li.get(f"{{{_STAREA_NS}}}h")
        if not (cx_t and cy_t and nw_t and nh_t):
            continue
        try:
            results.append({
                "index": float(idx),
                "cx": float(cx_t),
                "cy": float(cy_t),
                "nw": float(nw_t),
                "nh": float(nh_t),
            })
        except (TypeError, ValueError):
            pass
        idx += 1
    return results


# ---------------------------------------------------------------------------
# Derived number → region index mapping
# ---------------------------------------------------------------------------


def _derived_number_from_xmp(crop_xmp: Path) -> int | None:
    """Extract the derived number from a crop XMP filename, e.g. _D03-00_V.xmp → 3."""
    match = DERIVED_NAME_RE.search(crop_xmp.stem)
    if match is None:
        return None
    try:
        return int(match.group("derived"))
    except (KeyError, ValueError):
        return None


def _archive_max_derived(page_xmp: Path) -> int:
    """Return the highest D## used by archive-derived images for this page."""
    if not is_pages_dir(page_xmp.parent):
        return 0
    archive_dir = archive_dir_for_album_dir(page_xmp.parent)
    if not archive_dir.is_dir():
        return 0
    prefix = _page_prefix(page_xmp)
    derived_re = re.compile(rf"^{re.escape(prefix)}_D(\d+)", re.IGNORECASE)
    highest = 0
    for p in archive_dir.iterdir():
        if p.suffix.lower() == ".xmp":
            continue
        m = derived_re.match(p.stem)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest


def _crop_region_in_page(
    crop_xmp: Path,
    page_xmp: Path,
    page_regions: list[dict[str, float]],
    archive_max: int,
) -> dict[str, float] | None:
    """Return the page-space (cx, cy, nw, nh) for a crop's bounding box, or None."""
    derived_n = _derived_number_from_xmp(crop_xmp)
    if derived_n is None:
        return None
    region_1based = derived_n - archive_max
    if region_1based < 1:
        log.debug("Crop %s derived_n=%d archive_max=%d → invalid region index", crop_xmp.name, derived_n, archive_max)
        return None
    idx_0based = region_1based - 1
    if idx_0based >= len(page_regions):
        log.debug("Crop %s region index %d out of range (%d regions)", crop_xmp.name, idx_0based, len(page_regions))
        return None
    return page_regions[idx_0based]


# ---------------------------------------------------------------------------
# Coordinate projections
# ---------------------------------------------------------------------------


def _crop_face_to_page(
    face: dict[str, Any],
    region: dict[str, float],
) -> tuple[float, float, float, float]:
    """Project a face from crop space to page space using the crop's mwg-rs region box.

    region: {cx, cy, nw, nh} — center and size of the crop in page normalized coords.
    face: {rx, ry, rw, rh} — face in crop-normalized coords (0–1).
    Returns (page_rx, page_ry, page_rw, page_rh).
    """
    ox = region["cx"] - region["nw"] / 2.0  # crop's left in page normalized
    oy = region["cy"] - region["nh"] / 2.0  # crop's top in page normalized
    nw = region["nw"]
    nh = region["nh"]

    page_rx = ox + face["rx"] * nw
    page_ry = oy + face["ry"] * nh
    page_rw = face["rw"] * nw
    page_rh = face["rh"] * nh
    return page_rx, page_ry, page_rw, page_rh


def _page_face_to_crop(
    page_rx: float,
    page_ry: float,
    page_rw: float,
    page_rh: float,
    region: dict[str, float],
) -> tuple[float, float, float, float]:
    """Inverse of _crop_face_to_page: project page-space box back into crop space."""
    ox = region["cx"] - region["nw"] / 2.0
    oy = region["cy"] - region["nh"] / 2.0
    nw = region["nw"]
    nh = region["nh"]

    crop_rx = (page_rx - ox) / nw if nw > 0 else 0.0
    crop_ry = (page_ry - oy) / nh if nh > 0 else 0.0
    crop_rw = page_rw / nw if nw > 0 else 0.0
    crop_rh = page_rh / nh if nh > 0 else 0.0
    return crop_rx, crop_ry, crop_rw, crop_rh


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Multi-scan detection
# ---------------------------------------------------------------------------


def _scan_count_for_page(page_xmp: Path) -> int:
    """Return the number of TIF scans for the page (1 = single, >1 = stitched/multi)."""
    archive_dir = archive_dir_for_album_dir(page_xmp.parent) if is_pages_dir(page_xmp.parent) else None
    if archive_dir is None or not archive_dir.is_dir():
        return 1
    _, _, _, page_str = parse_album_filename(page_xmp.name)
    if not page_str.isdigit():
        return 1
    page_int = int(page_str)
    return sum(
        1 for p in archive_dir.iterdir()
        if SCAN_TIFF_RE.match(p.name) and int(SCAN_TIFF_RE.match(p.name).group("page")) == page_int  # type: ignore[union-attr]
    )


# ---------------------------------------------------------------------------
# Project all sources into page space
# ---------------------------------------------------------------------------


def project_sources_to_page(
    page_xmp: Path,
    scan_xmps: list[Path],
    crop_xmps: list[Path],
) -> tuple[list[ProjectedFaceBox], list[SourceFaceBox]]:
    """Project face boxes from all sources into page coordinate space.

    Returns (projected, unresolved).
    - projected: list of ProjectedFaceBox ready for clustering
    - unresolved: SourceFaceBox entries that could not be projected
    """
    projected: list[ProjectedFaceBox] = []
    unresolved: list[SourceFaceBox] = []

    # --- Page faces (already in page space) ---
    for fb in read_iptc_face_boxes(page_xmp):
        src = SourceFaceBox(
            name=fb["name"],
            rx=fb["rx"], ry=fb["ry"], rw=fb["rw"], rh=fb["rh"],
            source_kind="page",
            source_xmp=page_xmp,
        )
        projected.append(ProjectedFaceBox(
            name=src.name,
            page_rx=src.rx, page_ry=src.ry, page_rw=src.rw, page_rh=src.rh,
            confidence="high",
            origin=src,
        ))

    # --- Crop faces ---
    page_regions = read_page_photo_regions(page_xmp)
    archive_max = _archive_max_derived(page_xmp)

    for crop_xmp in crop_xmps:
        region = _crop_region_in_page(crop_xmp, page_xmp, page_regions, archive_max)
        for fb in read_iptc_face_boxes(crop_xmp):
            src = SourceFaceBox(
                name=fb["name"],
                rx=fb["rx"], ry=fb["ry"], rw=fb["rw"], rh=fb["rh"],
                source_kind="crop",
                source_xmp=crop_xmp,
            )
            if region is None:
                log.debug("Cannot map crop %s to page region; marking unresolved", crop_xmp.name)
                unresolved.append(src)
                continue
            page_rx, page_ry, page_rw, page_rh = _crop_face_to_page(fb, region)
            projected.append(ProjectedFaceBox(
                name=src.name,
                page_rx=page_rx, page_ry=page_ry, page_rw=page_rw, page_rh=page_rh,
                confidence="high",
                origin=src,
            ))

    # --- Archive scan faces ---
    scan_count = _scan_count_for_page(page_xmp)
    scan_confidence = "approximate" if scan_count == 1 else "unresolved"

    for scan_xmp in scan_xmps:
        for fb in read_iptc_face_boxes(scan_xmp):
            src = SourceFaceBox(
                name=fb["name"],
                rx=fb["rx"], ry=fb["ry"], rw=fb["rw"], rh=fb["rh"],
                source_kind="scan",
                source_xmp=scan_xmp,
            )
            if scan_confidence == "unresolved":
                log.debug(
                    "Multi-scan page (%d scans) — cannot project scan face to page space: %s",
                    scan_count, scan_xmp.name,
                )
                unresolved.append(src)
                continue
            # Single scan: treat normalized scan coords ≈ page coords (approximate)
            projected.append(ProjectedFaceBox(
                name=src.name,
                page_rx=src.rx, page_ry=src.ry, page_rw=src.rw, page_rh=src.rh,
                confidence="approximate",
                origin=src,
            ))

    return projected, unresolved


# ---------------------------------------------------------------------------
# IoU-based clustering
# ---------------------------------------------------------------------------


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _box(pf: ProjectedFaceBox) -> tuple[float, float, float, float]:
    return pf.page_rx, pf.page_ry, pf.page_rw, pf.page_rh


def _cluster_box(cluster_sources: list[ProjectedFaceBox]) -> tuple[float, float, float, float]:
    """Choose the canonical box for a cluster.

    Priority: prefer a "high" confidence page source, then any "high" source,
    then approximate, then centroid of all.
    """
    page_high = [s for s in cluster_sources if s.confidence == "high" and s.origin.source_kind == "page"]
    if page_high:
        return _box(page_high[0])
    high = [s for s in cluster_sources if s.confidence == "high"]
    if high:
        return _box(high[0])
    approx = [s for s in cluster_sources if s.confidence == "approximate"]
    if approx:
        return _box(approx[0])
    # Centroid fallback
    rxs = [s.page_rx for s in cluster_sources]
    rys = [s.page_ry for s in cluster_sources]
    rws = [s.page_rw for s in cluster_sources]
    rhs = [s.page_rh for s in cluster_sources]
    n = len(cluster_sources)
    return sum(rxs) / n, sum(rys) / n, sum(rws) / n, sum(rhs) / n


def cluster_face_boxes(
    projected: list[ProjectedFaceBox],
    *,
    iou_threshold: float = _DEFAULT_IOU_THRESHOLD,
) -> list[FaceCluster]:
    """Greedy IoU-based clustering of projected face boxes.

    Boxes that overlap above iou_threshold are merged into the same cluster.
    """
    # Build clusters as lists of ProjectedFaceBox
    raw_clusters: list[list[ProjectedFaceBox]] = []
    raw_boxes: list[tuple[float, float, float, float]] = []  # cluster representative box

    for pf in projected:
        box = _box(pf)
        best_idx = -1
        best_iou = iou_threshold
        for i, cluster_box in enumerate(raw_boxes):
            score = _iou(box, cluster_box)
            if score > best_iou:
                best_iou = score
                best_idx = i

        if best_idx >= 0:
            raw_clusters[best_idx].append(pf)
            # Update representative to the chosen canonical box of the merged set
            raw_boxes[best_idx] = _cluster_box(raw_clusters[best_idx])
        else:
            raw_clusters.append([pf])
            raw_boxes.append(box)

    clusters: list[FaceCluster] = []
    for sources in raw_clusters:
        rx, ry, rw, rh = _cluster_box(sources)
        all_names = [str(s.name or "").strip() for s in sources]
        distinct_names = list(dict.fromkeys(n for n in all_names if n))
        has_conflict = len(distinct_names) > 1
        clusters.append(FaceCluster(
            resolved_rx=rx,
            resolved_ry=ry,
            resolved_rw=rw,
            resolved_rh=rh,
            names=distinct_names,
            sources=sources,
            has_conflict=has_conflict,
        ))
    return clusters


# ---------------------------------------------------------------------------
# Backfill planning: decide what to write where
# ---------------------------------------------------------------------------


def _box_already_covered(
    rx: float, ry: float, rw: float, rh: float,
    existing_faces: list[dict[str, Any]],
    *,
    iou_threshold: float = _DEFAULT_IOU_THRESHOLD,
) -> bool:
    """Return True if an equivalent face box already exists in the source XMP."""
    candidate = (rx, ry, rw, rh)
    for ef in existing_faces:
        existing = (ef["rx"], ef["ry"], ef["rw"], ef["rh"])
        if _iou(candidate, existing) > iou_threshold:
            return True
    return False


def _source_already_in_cluster(cluster: FaceCluster, xmp_path: Path) -> bool:
    """Return True if xmp_path already contributed a face to this cluster."""
    return any(s.origin.source_xmp == xmp_path for s in cluster.sources)


def plan_backfill(
    page_xmp: Path,
    clusters: list[FaceCluster],
    scan_xmps: list[Path],
    crop_xmps: list[Path],
    page_regions: list[dict[str, float]],
    archive_max: int,
    *,
    skip_conflicts: bool = True,
    iou_threshold: float = _DEFAULT_IOU_THRESHOLD,
) -> list[PendingWrite]:
    """Determine which face boxes need to be written to which source XMPs.

    For each cluster, for each source that is missing that face:
    - page XMP gets the resolved cluster box
    - crop XMPs get the box transformed back into crop space
    - scan XMPs (single-scan) get the box as-is (approximate)

    If skip_conflicts=True, clusters with multiple distinct names are skipped.
    """
    pending: list[PendingWrite] = []
    scan_count = _scan_count_for_page(page_xmp)

    for cluster in clusters:
        if skip_conflicts and cluster.has_conflict:
            log.debug(
                "Skipping backfill for conflicted cluster (names: %s)", cluster.names
            )
            continue
        resolved_name = cluster.names[0] if cluster.names else ""
        rx, ry, rw, rh = cluster.resolved_rx, cluster.resolved_ry, cluster.resolved_rw, cluster.resolved_rh

        # --- Page XMP ---
        if not _source_already_in_cluster(cluster, page_xmp):
            existing = read_iptc_face_boxes(page_xmp)
            if not _box_already_covered(rx, ry, rw, rh, existing, iou_threshold=iou_threshold):
                pending.append(PendingWrite(
                    xmp_path=page_xmp,
                    source_kind="page",
                    name=resolved_name,
                    rx=rx, ry=ry, rw=rw, rh=rh,
                    reason="face cluster not present in page XMP",
                ))

        # --- Crop XMPs ---
        for crop_xmp in crop_xmps:
            if _source_already_in_cluster(cluster, crop_xmp):
                continue
            region = _crop_region_in_page(crop_xmp, page_xmp, page_regions, archive_max)
            if region is None:
                continue
            # Check if the cluster box overlaps the crop's page-space region
            crop_ox = region["cx"] - region["nw"] / 2.0
            crop_oy = region["cy"] - region["nh"] / 2.0
            crop_box_page = (crop_ox, crop_oy, region["nw"], region["nh"])
            if _iou((rx, ry, rw, rh), crop_box_page) < 0.05:
                continue  # face doesn't overlap this crop
            crx, cry, crw, crh = _page_face_to_crop(rx, ry, rw, rh, region)
            crx, cry = _clamp01(crx), _clamp01(cry)
            crw = max(0.01, min(1.0 - crx, crw))
            crh = max(0.01, min(1.0 - cry, crh))
            existing = read_iptc_face_boxes(crop_xmp)
            if not _box_already_covered(crx, cry, crw, crh, existing, iou_threshold=iou_threshold):
                pending.append(PendingWrite(
                    xmp_path=crop_xmp,
                    source_kind="crop",
                    name=resolved_name,
                    rx=crx, ry=cry, rw=crw, rh=crh,
                    reason="face cluster not present in crop XMP",
                ))

        # --- Scan XMPs (single-scan only; multi-scan = unresolved, skip) ---
        if scan_count > 1:
            continue
        for scan_xmp in scan_xmps:
            if _source_already_in_cluster(cluster, scan_xmp):
                continue
            existing = read_iptc_face_boxes(scan_xmp)
            if not _box_already_covered(rx, ry, rw, rh, existing, iou_threshold=iou_threshold):
                pending.append(PendingWrite(
                    xmp_path=scan_xmp,
                    source_kind="scan",
                    name=resolved_name,
                    rx=rx, ry=ry, rw=rw, rh=rh,
                    reason="face cluster not present in scan XMP (approximate: single-scan page)",
                ))

    return pending


# ---------------------------------------------------------------------------
# IPTC face region write-back (merge into existing XMP)
# ---------------------------------------------------------------------------

for _prefix, _uri in [
    ("x", _X_NS),
    ("rdf", _RDF_NS),
    ("Iptc4xmpExt", _IPTC_EXT_NS),
]:
    ET.register_namespace(_prefix, _uri)


def _get_or_create_rdf_desc(tree: ET.ElementTree) -> ET.Element:
    root = tree.getroot()
    assert root is not None
    rdf = root.find(_RDF_ROOT)
    if rdf is None:
        rdf = ET.SubElement(root, _RDF_ROOT)
    desc = rdf.find(_RDF_DESC)
    if desc is None:
        desc = ET.SubElement(rdf, _RDF_DESC)
        desc.set(f"{{{_RDF_NS}}}about", "")
    return desc


def _build_minimal_xmp_tree() -> ET.ElementTree:
    xmpmeta = ET.Element(f"{{{_X_NS}}}xmpmeta")
    rdf = ET.SubElement(xmpmeta, _RDF_ROOT)
    desc = ET.SubElement(rdf, _RDF_DESC)
    desc.set(f"{{{_RDF_NS}}}about", "")
    return ET.ElementTree(xmpmeta)


def _add_iptc_face_li(bag: ET.Element, face_n: int, name: str, rx: float, ry: float, rw: float, rh: float) -> None:
    li = ET.SubElement(bag, _RDF_LI)
    li.set(_RDF_PARSE_TYPE, "Resource")
    boundary = ET.SubElement(li, _IPTC_REGION_BOUNDARY)
    boundary.set(_RDF_PARSE_TYPE, "Resource")
    ET.SubElement(boundary, _IPTC_RB_SHAPE).text = "rectangle"
    ET.SubElement(boundary, _IPTC_RB_UNIT).text = "relative"
    ET.SubElement(boundary, _IPTC_RB_X).text = f"{rx:.6f}"
    ET.SubElement(boundary, _IPTC_RB_Y).text = f"{ry:.6f}"
    ET.SubElement(boundary, _IPTC_RB_W).text = f"{rw:.6f}"
    ET.SubElement(boundary, _IPTC_RB_H).text = f"{rh:.6f}"
    ET.SubElement(li, _IPTC_RCTYPE).text = "face-identified"
    ET.SubElement(li, _IPTC_RID).text = f"face-{face_n}"
    if name:
        name_el = ET.SubElement(li, _IPTC_NAME)
        alt = ET.SubElement(name_el, _RDF_ALT)
        li_name = ET.SubElement(alt, _RDF_LI)
        li_name.set(_XML_LANG, "x-default")
        li_name.text = name


def merge_iptc_face_box(
    xmp_path: Path,
    name: str,
    rx: float,
    ry: float,
    rw: float,
    rh: float,
    *,
    iou_threshold: float = _DEFAULT_IOU_THRESHOLD,
) -> bool:
    """Add a single IPTC face region to an XMP sidecar if not already present.

    Skips the write if an overlapping face box already exists (by IoU).
    Returns True if a write occurred.
    """
    existing = read_iptc_face_boxes(xmp_path)
    if _box_already_covered(rx, ry, rw, rh, existing, iou_threshold=iou_threshold):
        return False

    if xmp_path.is_file():
        try:
            tree: ET.ElementTree = ET.parse(str(xmp_path))  # type: ignore[assignment]
        except ET.ParseError:
            tree = _build_minimal_xmp_tree()
    else:
        tree = _build_minimal_xmp_tree()

    desc = _get_or_create_rdf_desc(tree)

    ir_el = desc.find(_IPTC_IMAGE_REGION)
    if ir_el is None:
        ir_el = ET.SubElement(desc, _IPTC_IMAGE_REGION)
    bag = ir_el.find(_RDF_BAG)
    if bag is None:
        bag = ET.SubElement(ir_el, _RDF_BAG)

    # Count existing face-li items to assign next face-N id
    face_n = sum(1 for li in bag.findall(_RDF_LI) if _is_face_li(li)) + 1
    _add_iptc_face_li(bag, face_n, name, rx, ry, rw, rh)

    ET.indent(tree, space="  ")
    xmp_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(xmp_path), encoding="utf-8", xml_declaration=True)
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def reconcile_page(
    page_xmp: Path,
    *,
    dry_run: bool = True,
    iou_threshold: float = _DEFAULT_IOU_THRESHOLD,
    skip_conflicts: bool = True,
) -> ReconcileResult:
    """Reconcile face regions across all sources belonging to a page.

    Steps:
    1. Collect page XMP, archive scan XMPs, crop XMPs.
    2. Parse IPTC face regions from each source.
    3. Project all faces into page coordinate space.
    4. Cluster overlapping boxes by IoU.
    5. Choose a resolved canonical box per cluster.
    6. Plan backfill writes for sources missing each cluster.
    7. Execute writes unless dry_run=True.

    Args:
        page_xmp: Path to the page view XMP sidecar (_P##_V.xmp).
        dry_run: If True, compute and return pending writes without executing them.
        iou_threshold: IoU threshold for merging face boxes into clusters.
        skip_conflicts: If True, skip backfill for clusters with conflicting names.

    Returns:
        ReconcileResult with clusters, conflicts, unresolved sources, and pending writes.
    """
    page_xmp = Path(page_xmp)
    scan_xmps = find_archive_scans_for_page(page_xmp)
    crop_xmps = find_crop_xmps_for_page(page_xmp)
    sources_scanned = [page_xmp] + scan_xmps + crop_xmps

    projected, unresolved = project_sources_to_page(page_xmp, scan_xmps, crop_xmps)
    clusters = cluster_face_boxes(projected, iou_threshold=iou_threshold)
    conflicts = [c for c in clusters if c.has_conflict]

    page_regions = read_page_photo_regions(page_xmp)
    archive_max = _archive_max_derived(page_xmp)

    pending_writes = plan_backfill(
        page_xmp,
        clusters,
        scan_xmps,
        crop_xmps,
        page_regions,
        archive_max,
        skip_conflicts=skip_conflicts,
        iou_threshold=iou_threshold,
    )

    if not dry_run:
        for pw in pending_writes:
            try:
                wrote = merge_iptc_face_box(
                    pw.xmp_path,
                    pw.name,
                    pw.rx, pw.ry, pw.rw, pw.rh,
                    iou_threshold=iou_threshold,
                )
                if wrote:
                    log.info(
                        "Wrote face '%s' → %s (%s)", pw.name, pw.xmp_path.name, pw.source_kind
                    )
            except Exception:
                log.exception("Failed to write face region to %s", pw.xmp_path)

    return ReconcileResult(
        page_xmp=page_xmp,
        clusters=clusters,
        conflicts=conflicts,
        unresolved_sources=unresolved,
        pending_writes=pending_writes,
        sources_scanned=sources_scanned,
    )
