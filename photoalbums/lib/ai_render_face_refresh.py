from __future__ import annotations

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

        effective, settings_sig, creator_tool, _date_estimation_enabled = self.runner._resolve_effective_settings(
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
            creator_tool,
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
        if force:
            clear_pipeline_steps(sidecar, ["face_refresh"])
        else:
            pipeline_state = read_pipeline_step(sidecar, "face_refresh")
            if pipeline_state is not None:
                return False

        image = Path(image_path)
        self._refresh_with_runner(image, sidecar)
        write_pipeline_step(sidecar, "face_refresh", model="buffalo_l")
        return True


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
