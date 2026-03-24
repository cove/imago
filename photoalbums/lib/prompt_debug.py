from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PromptDebugStep:
    step: str
    engine: str
    model: str
    prompt: str
    system_prompt: str = ""
    source_path: str = ""
    prompt_source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": str(self.step or "").strip(),
            "engine": str(self.engine or "").strip(),
            "model": str(self.model or "").strip(),
            "prompt": str(self.prompt or ""),
            "system_prompt": str(self.system_prompt or ""),
            "source_path": str(self.source_path or "").strip(),
            "prompt_source": str(self.prompt_source or "").strip(),
            "metadata": dict(self.metadata or {}),
        }


class PromptDebugSession:
    def __init__(self, image_path: str | Path, *, label: str = "") -> None:
        self.image_path = Path(image_path)
        self.label = str(label or "").strip()
        self._steps: list[PromptDebugStep] = []

    def record(
        self,
        *,
        step: str,
        engine: str,
        model: str,
        prompt: str,
        system_prompt: str = "",
        source_path: str | Path | None = None,
        prompt_source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not str(prompt or "").strip() and not str(system_prompt or "").strip():
            return
        self._steps.append(
            PromptDebugStep(
                step=str(step or "").strip(),
                engine=str(engine or "").strip(),
                model=str(model or "").strip(),
                prompt=str(prompt or ""),
                system_prompt=str(system_prompt or ""),
                source_path=str(source_path or "").strip(),
                prompt_source=str(prompt_source or "").strip(),
                metadata=dict(metadata or {}),
            )
        )

    def has_steps(self) -> bool:
        return bool(self._steps)

    def to_artifact(self) -> dict[str, Any]:
        return {
            "kind": "photoalbums_prompts",
            "image_path": str(self.image_path),
            "label": self.label or self.image_path.name,
            "step_count": len(self._steps),
            "steps": [row.to_dict() for row in self._steps],
        }
