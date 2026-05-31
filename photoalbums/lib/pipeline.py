from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class PipelineStep:
    id: str
    label: str
    depends_on: list[str]
    optional: bool = False


def _make_steps() -> list[PipelineStep]:
    return [
        PipelineStep(
            id="render",
            label="Stitch/convert archive scans to page view JPEGs",
            depends_on=[],
        ),
        PipelineStep(
            id="propagate-scan-context",
            label="Copy OCR text and location context from scan XMP to view XMP",
            depends_on=["render"],
        ),
        PipelineStep(
            id="detect-regions",
            label="Detect photo bounding boxes and write MWG-RS XMP regions",
            depends_on=["render"],
        ),
        PipelineStep(
            id="crop-regions",
            label="Crop detected regions to _Photos/ directory",
            depends_on=["detect-regions"],
        ),
        PipelineStep(
            id="face-refresh",
            label="Update face region metadata on rendered outputs",
            depends_on=["crop-regions"],
        ),
        PipelineStep(
            id="ocr",
            label="Run OCR on rendered view JPEG",
            depends_on=["propagate-scan-context"],
        ),
        PipelineStep(
            id="ai-index",
            label="Run AI pipeline (caption, GPS, XMP write)",
            depends_on=["ocr"],
        ),
        PipelineStep(
            id="propagate-to-crops",
            label="Propagate location, GPS, and person names from page XMP to crop XMPs",
            depends_on=["ai-index"],
        ),
        PipelineStep(
            id="immich-face-refresh",
            label="Refresh face region metadata from current Immich assets",
            depends_on=["ai-index", "face-refresh"],
        ),
        PipelineStep(
            id="face-reconcile",
            label="Reconcile and backfill IPTC face boxes across page, scan, and crop XMPs",
            depends_on=["immich-face-refresh"],
        ),
        PipelineStep(
            id="sequence-page-dates",
            label="Slew _Archive/_Pages dates monotonically across album date anchors",
            depends_on=["ai-index"],
        ),
        PipelineStep(
            id="verify-crops",
            label="Review each page's crops against the page image and page/crop XMP context",
            depends_on=["propagate-to-crops"],
            optional=True,
        ),
    ]


PIPELINE_STEPS: list[PipelineStep] = _make_steps()

VALID_STEP_IDS: list[str] = [s.id for s in PIPELINE_STEPS]

OPTIONAL_STEP_IDS: set[str] = {s.id for s in PIPELINE_STEPS if s.optional}


def _dag_node_label(step: PipelineStep, number: dict[str, int], via: str | None) -> str:
    tag = f"[{number[step.id]}] "
    opt = " [optional]" if step.optional else ""
    others = [d for d in step.depends_on if d != via]
    extra = f"  (also after: {', '.join(f'[{number[d]}] {d}' for d in others)})" if others else ""
    return f"{tag}{step.id}{opt}{extra}"


def _emit_dag_node(
    sid: str,
    *,
    prefix: str,
    is_last: bool,
    via: str | None,
    by_id: dict[str, PipelineStep],
    number: dict[str, int],
    children: dict[str, list[str]],
    expanded: set[str],
    lines: list[str],
) -> None:
    if via is None:
        lines.append(_dag_node_label(by_id[sid], number, via))
        child_prefix = ""
    else:
        connector = "└─ " if is_last else "├─ "
        if sid in expanded and children[sid]:
            lines.append(f"{prefix}{connector}[{number[sid]}] {sid}  ↑ (shown above)")
            return
        lines.append(f"{prefix}{connector}{_dag_node_label(by_id[sid], number, via)}")
        child_prefix = prefix + ("   " if is_last else "│  ")
    expanded.add(sid)
    kids = children[sid]
    for idx, kid in enumerate(kids):
        _emit_dag_node(
            kid,
            prefix=child_prefix,
            is_last=idx == len(kids) - 1,
            via=sid,
            by_id=by_id,
            number=number,
            children=children,
            expanded=expanded,
            lines=lines,
        )


def format_pipeline_dag(steps: list[PipelineStep]) -> list[str]:
    """Render the dependency DAG as an indented ASCII tree.

    Execution order is the registration order of ``steps``; this view instead
    exposes the ``depends_on`` edges. The graph is a DAG (not a tree): a node
    with more than one parent is expanded in full the first time it appears and
    shown as a back-reference (``↑ (shown above)``) under its other parents.
    Extra parents are annotated inline as ``(also after: ...)``. Each node is
    prefixed with its ``[N]`` execution-order number.
    """
    by_id = {s.id: s for s in steps}
    number = {s.id: i for i, s in enumerate(steps, 1)}
    children: dict[str, list[str]] = {s.id: [] for s in steps}
    for step in steps:
        for dep in step.depends_on:
            if dep in children:
                children[dep].append(step.id)
    lines: list[str] = []
    expanded: set[str] = set()
    for root in (s.id for s in steps if not s.depends_on):
        _emit_dag_node(
            root,
            prefix="",
            is_last=True,
            via=None,
            by_id=by_id,
            number=number,
            children=children,
            expanded=expanded,
            lines=lines,
        )
    return lines


def validate_step_ids(ids: list[str], *, flag: str) -> list[str]:
    """Validate step ids against the registry; print error and exit 2 on unknown ids."""
    unknown = [sid for sid in ids if sid not in VALID_STEP_IDS]
    if unknown:
        print(
            f"Error: unknown step id(s) for {flag}: {', '.join(unknown)}\nValid step ids: {', '.join(VALID_STEP_IDS)}",
            file=sys.stderr,
        )
        sys.exit(2)
    return ids
