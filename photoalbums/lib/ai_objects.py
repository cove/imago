from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .model_store import YOLO_MODEL_DIR


def _resolve_model_reference(model_name: str) -> tuple[str, Path | None]:
    text = str(model_name or "").strip()
    if not text:
        text = "models/yolo11n.pt"

    path = Path(text).expanduser()
    # Keep explicit paths unchanged so callers can still opt into custom models.
    if path.is_absolute() or any(part not in {"", "."} for part in path.parts[:-1]):
        return str(path), None

    model_file = path.name
    if model_file.lower().endswith(".pt"):
        YOLO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return model_file, YOLO_MODEL_DIR
    return text, None


@contextmanager
def _pushd(path: Path):
    current = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(current)


@dataclass
class ObjectDetection:
    label: str
    score: float


class YOLOObjectDetector:
    def __init__(
        self,
        *,
        model_name: str = "models/yolo11n.pt",
        confidence: float = 0.30,
        max_detections: int = 100,
    ):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - dependency optional
            raise RuntimeError(
                "Ultralytics is required for object detection. Install with: pip install ultralytics"
            ) from exc

        model_ref, model_dir = _resolve_model_reference(model_name)
        if model_dir is None:
            self._model = YOLO(model_ref)
        else:
            # When downloading stock YOLO weights, keep them under repo-root models/.
            with _pushd(model_dir):
                self._model = YOLO(model_ref)
        self.confidence = float(confidence)
        self.max_detections = int(max_detections)

    def detect_image(self, image_path: str | Path) -> list[ObjectDetection]:
        import cv2

        img = cv2.imread(str(image_path))
        if img is not None and (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        source = img if img is not None else str(image_path)
        results = self._model.predict(
            source=source,
            conf=self.confidence,
            max_det=self.max_detections,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = getattr(result, "names", {}) or {}
        labels_by_name: dict[str, float] = {}

        cls_vals = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
        conf_vals = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
        for idx, raw in enumerate(cls_vals):
            label = str(names.get(int(raw), int(raw)))
            score = float(conf_vals[idx]) if idx < len(conf_vals) else 0.0
            current = labels_by_name.get(label)
            if current is None or score > current:
                labels_by_name[label] = score

        out = [ObjectDetection(label=label, score=score) for label, score in labels_by_name.items()]
        out.sort(key=lambda row: row.score, reverse=True)
        return out
