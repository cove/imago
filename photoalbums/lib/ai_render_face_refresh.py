from __future__ import annotations

import re
from pathlib import Path

from .ai_index_runner import IndexRunner
from .ai_sidecar_state import has_valid_sidecar
from .xmp_sidecar import (
    clear_pipeline_steps,
    read_ai_sidecar_state,
    read_person_in_image,
    read_pipeline_step,
    write_pipeline_step,
)

_DERIVED_VIEW_RE = re.compile(r"^(?P<page>.+_P\d{2})_D\d{2}-\d{2}_V\.jpg$")


class FaceRefreshSkipped(RuntimeError):
    pass


class RenderFaceRefreshSession:
    def __init__(self, *, photos_root: str | Path) -> None:
        self.photos_root = Path(photos_root).expanduser().resolve(strict=False)
        self.runner = IndexRunner(["--photos-root", str(self.photos_root), "--include-view"])
        self.runner.files = []

    def set_files(self, files: list[Path]) -> None:
        self.runner.files = [Path(path) for path in files]

    def _refresh_with_runner(self, image_path: Path, sidecar_path: Path) -> None:
        if not has_valid_sidecar(image_path):
            raise RuntimeError(f"Rendered sidecar missing or invalid for face refresh: {sidecar_path}")
        existing_sidecar_state = read_ai_sidecar_state(sidecar_path)
        if not isinstance(existing_sidecar_state, dict):
            raise RuntimeError(f"Rendered sidecar could not be parsed for face refresh: {sidecar_path}")

        effective, settings_sig, _date_estimation_enabled = self.runner._resolve_effective_settings(
            image_path
        )
        try:
            people_matcher, current_cast_signature = self.runner._get_people_matcher_and_signature(effective)
        except Exception as exc:
            raise FaceRefreshSkipped(f"face refresh skipped for {image_path.name}: {exc}") from exc
        if people_matcher is None:
            raise FaceRefreshSkipped(f"face refresh skipped for {image_path.name}: people matching disabled")

        try:
            index = self.runner.files.index(image_path) + 1
        except ValueError:
            self.runner.files = [image_path]
            index = 1

        self.runner._process_people_update(
            index,
            image_path,
            sidecar_path,
            effective,
            settings_sig,
            False,
            existing_sidecar_state,
            read_person_in_image(sidecar_path),
            people_matcher,
            current_cast_signature,
            False,
            preserve_existing_xmp_people=False,
            raise_on_error=True,
        )

    def refresh_face_regions(
        self,
        image_path: str | Path,
        sidecar_path: str | Path,
        *,
        force: bool = False,
    ) -> bool:
        sidecar = Path(sidecar_path)
        image = Path(image_path)
        if force:
            clear_pipeline_steps(sidecar, ["face_refresh"])
        else:
            pipeline_state = read_pipeline_step(sidecar, "face_refresh")
            if pipeline_state is not None:
                return False
            legacy_page_state = read_pipeline_step(sidecar, "face-refresh")
            if legacy_page_state is not None and self._is_page_view_target(image):
                return False
            if not self._has_face_refresh_work(image, sidecar):
                return False

        self._refresh_with_runner(image, sidecar)
        write_pipeline_step(sidecar, "face_refresh", model="buffalo_l")
        return True

    @staticmethod
    def _is_page_view_target(image_path: Path) -> bool:
        name = image_path.name
        return name.endswith("_V.jpg") and re.search(r"_D\d{2}-\d{2}_V\.jpg$", name) is None

    @classmethod
    def _has_face_refresh_work(cls, image_path: Path, sidecar_path: Path) -> bool:
        signal = cls._face_refresh_signal(sidecar_path)
        if signal is not None:
            return signal
        source_page_sidecar = cls._source_page_sidecar(image_path)
        if source_page_sidecar is None or source_page_sidecar == sidecar_path:
            return True
        source_signal = cls._face_refresh_signal(source_page_sidecar)
        return source_signal is not False

    @staticmethod
    def _face_refresh_signal(sidecar_path: Path) -> bool | None:
        if not sidecar_path.is_file():
            return None
        state = read_ai_sidecar_state(sidecar_path)
        if not isinstance(state, dict):
            return None
        if read_person_in_image(sidecar_path):
            return True
        if state.get("people_detected") is True or state.get("people_identified") is True:
            return True
        detections = state.get("detections")
        if isinstance(detections, dict):
            people = detections.get("people")
            if isinstance(people, list) and any(isinstance(row, dict) for row in people):
                return True
        if state.get("people_detected") is False and state.get("people_identified") is not True:
            return False
        return None

    @staticmethod
    def _source_page_sidecar(image_path: Path) -> Path | None:
        match = _DERIVED_VIEW_RE.match(image_path.name)
        if match is None:
            return None
        page_sidecar_name = f"{match.group('page')}_V.xmp"
        parent = image_path.parent
        if parent.name.endswith("_Photos"):
            pages_dir = parent.with_name(f"{parent.name[:-len('_Photos')]}_Pages")
            return pages_dir / page_sidecar_name
        return parent / page_sidecar_name


def refresh_face_regions(
    image_path: str | Path,
    sidecar_path: str | Path,
    *,
    force: bool = False,
) -> bool:
    image = Path(image_path).expanduser().resolve(strict=False)
    session = RenderFaceRefreshSession(photos_root=image.parent.parent)
    session.set_files([image])
    return session.refresh_face_regions(image, sidecar_path, force=force)
