from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Suppress FutureWarnings from insightface internals (third-party library noise).
warnings.filterwarnings(
    "ignore",
    message=r"`rcond` parameter will change",
    category=FutureWarning,
    module=r"insightface",
)
warnings.filterwarnings(
    "ignore",
    message=r"`estimate` is deprecated",
    category=FutureWarning,
    module=r"insightface",
)

from .storage import TextFaceStore

CURRENT_FACE_EMBEDDING_MODEL = "insightface.buffalo_l.arcface_512"
CURRENT_FACE_DETECTOR_MODEL = "insightface.buffalo_l.detector"
FALLBACK_FACE_EMBEDDING_MODEL = "imago.simple_grayscale_32_v1"
FALLBACK_FACE_DETECTOR_MODEL = "opencv.haar_frontalface_default"


def _timestamp_from_seconds(seconds: float) -> str:
    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    whole_seconds, millis = divmod(rem, 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(whole_seconds):02d}.{int(millis):03d}"


def _expand_box(
    x: int, y: int, w: int, h: int, width: int, height: int, margin: float = 0.22
) -> tuple[int, int, int, int]:
    grow_w = int(round(float(w) * float(margin)))
    grow_h = int(round(float(h) * float(margin)))
    x0 = max(0, int(x) - grow_w)
    y0 = max(0, int(y) - grow_h)
    x1 = min(int(width), int(x + w) + grow_w)
    y1 = min(int(height), int(y + h) + grow_h)
    return x0, y0, x1, y1


def compute_simple_embedding(face_bgr: np.ndarray, out_size: int = 32) -> list[float]:
    if face_bgr is None or face_bgr.size == 0:
        return []
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(
        gray, (int(out_size), int(out_size)), interpolation=cv2.INTER_AREA
    )
    vec = resized.astype(np.float32).reshape(-1) / 255.0
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec = vec / norm
    return [float(item) for item in vec.tolist()]


_insightface_app = None
_insightface_app_error = ""


def _get_insightface_app() -> object | None:
    """Lazy-load InsightFace FaceAnalysis singleton. Downloads buffalo_l model on first call."""
    global _insightface_app, _insightface_app_error
    if _insightface_app is not None:
        return _insightface_app
    try:
        from insightface.app import FaceAnalysis  # type: ignore[import]

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_thresh=0.18, det_size=(640, 640))
        _insightface_app_error = ""
        _insightface_app = app
        return _insightface_app
    except Exception as exc:
        _insightface_app_error = f"{type(exc).__name__}: {exc}"
        return None


def insightface_load_error() -> str:
    if _insightface_app is None:
        _get_insightface_app()
    return str(_insightface_app_error or "").strip()


def compute_arcface_embedding(face_bgr: np.ndarray) -> list[float] | None:
    """
    Compute a 512-dim L2-normalized ArcFace embedding using InsightFace.
    Returns None if InsightFace is unavailable or no face is found in the crop.
    Falls back to compute_simple_embedding() via the caller.
    """
    if face_bgr is None or face_bgr.size == 0:
        return None
    try:
        app = _get_insightface_app()
        if app is None:
            return None
        faces = app.get(face_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0) or 0.0))
        emb = getattr(face, "embedding", None)
        if emb is None or len(emb) == 0:
            return None
        vec = np.asarray(emb, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = vec / norm
        return [float(v) for v in vec.tolist()]
    except Exception:
        return None


def estimate_face_quality(face_bgr: np.ndarray) -> float:
    if face_bgr is None or face_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    brightness = float(np.mean(gray))
    sharp_component = max(0.0, min(1.0, sharpness / 180.0))
    bright_component = max(0.0, min(1.0, brightness / 255.0))
    return round((0.7 * sharp_component) + (0.3 * bright_component), 4)


def crop_has_visual_detail(
    face_bgr: np.ndarray,
    *,
    min_std: float = 3.0,              # was 6.0
    min_dynamic_range: float = 12.0,   # was 18.0
    min_laplacian_var: float = 2.0,    # was 4.0
) -> bool:
    """Reject obvious flat-color false positives from Haar detections."""
    if face_bgr is None or face_bgr.size == 0:
        return False
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    std = float(np.std(gray))
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    dynamic_range = float(p95 - p5)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    if std < float(min_std):
        return False
    if dynamic_range < float(min_dynamic_range):
        return False
    if lap_var < float(min_laplacian_var):
        return False
    return True



class FaceIngestor:
    def __init__(self, store: TextFaceStore, *, require_primary_model: bool = False):
        self.store = store
        self.require_primary_model = bool(require_primary_model)
        cascades = Path(str(cv2.data.haarcascades))
        face_path = cascades / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(face_path))
        self._insightface = _get_insightface_app()
        self._insightface_score_threshold = 0.10
        if self._cascade.empty():
            raise RuntimeError(f"Unable to load face cascade: {face_path}")

    def runtime_status(self) -> dict[str, Any]:
        primary_available = self._insightface is not None
        load_error = insightface_load_error()
        if primary_available:
            message = "InsightFace buffalo_l is active for new ingest."
        elif self.require_primary_model:
            message = (
                "InsightFace buffalo_l is unavailable, so new ingest is blocked instead of "
                "falling back to OpenCV Haar cascades."
            )
        else:
            message = (
                "InsightFace buffalo_l is unavailable. Cast will fall back to OpenCV Haar "
                "cascades for ingest in this mode."
            )
        return {
            "primary_required": bool(self.require_primary_model),
            "primary_available": bool(primary_available),
            "can_ingest": bool(primary_available or not self.require_primary_model),
            "fallback_active": bool(not primary_available),
            "active_detector_model": (
                CURRENT_FACE_DETECTOR_MODEL
                if primary_available
                else FALLBACK_FACE_DETECTOR_MODEL
            ),
            "active_embedding_model": (
                CURRENT_FACE_EMBEDDING_MODEL
                if primary_available
                else FALLBACK_FACE_EMBEDDING_MODEL
            ),
            "load_error": load_error,
            "message": message,
        }

    def _ensure_primary_model_ready(self) -> None:
        if not self.require_primary_model or self._insightface is not None:
            return
        detail = insightface_load_error()
        message = (
            "InsightFace buffalo_l is unavailable. Cast ingest is configured to require the "
            "primary face model and will not silently fall back to OpenCV Haar cascades."
        )
        if detail:
            message = f"{message} Load error: {detail}"
        raise RuntimeError(message)

    def is_valid_face_crop(
        self,
        crop_bgr: np.ndarray
    ) -> bool:
        if not crop_has_visual_detail(crop_bgr):
            return False
        h, w = crop_bgr.shape[:2]
        if h < 24 or w < 24:
            return False
        aspect = float(w) / float(max(1, h))
        return 0.50 <= aspect <= 2.0

    def _detect(
        self, image_bgr: np.ndarray, *, min_size: int = 40
    ) -> list[tuple[int, int, int, int]]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        if self._insightface is not None:
            if h < 12 or w < 12:
                return []
            try:
                faces = list(self._insightface.get(image_bgr) or [])
            except Exception:
                return []
            out: list[tuple[int, int, int, int]] = []
            for face in faces:
                score = float(getattr(face, "det_score", 0.0) or 0.0)
                if score < self._insightface_score_threshold:
                    continue
                try:
                    bbox = np.asarray(getattr(face, "bbox"), dtype=np.float32).reshape(-1)
                except Exception:
                    continue
                if bbox.size < 4:
                    continue
                x = int(max(0, min(round(float(bbox[0])), w - 1)))
                y = int(max(0, min(round(float(bbox[1])), h - 1)))
                x2 = int(max(x + 1, min(round(float(bbox[2])), w)))
                y2 = int(max(y + 1, min(round(float(bbox[3])), h)))
                fw, fh = x2 - x, y2 - y
                if min(fw, fh) < max(1, int(min_size)):
                    continue
                out.append((x, y, fw, fh))
            return out
        # Haar cascade fallback
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        side = int(max(1, min_size))
        if side > min(h, w):
            return []
        try:
            raw = self._cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(side, side), flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except cv2.error:
            return []
        if raw is None or np.asarray(raw).size == 0:
            return []
        rows = np.asarray(raw)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        return [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rows if int(r[2]) > 0 and int(r[3]) > 0]

    def _save_crop(self, face_id: str, crop_bgr: np.ndarray) -> str:
        crops_dir = self.store.root_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_name = f"{str(face_id).strip()}.jpg"
        crop_path = crops_dir / crop_name
        ok = cv2.imwrite(str(crop_path), crop_bgr)
        if not ok:
            raise RuntimeError(f"Failed to write crop image: {crop_path}")
        rel = crop_path.relative_to(self.store.root_dir)
        return rel.as_posix()

    def ingest_photo(
        self,
        *,
        image_path: str | Path,
        source_path: str | None = None,
        min_size: int = 40,
        max_faces: int = 50,
    ) -> list[dict[str, Any]]:
        self._ensure_primary_model_ready()
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Photo file does not exist: {path}")
        if path.is_dir():
            raise IsADirectoryError(
                f"Photo path is a directory. Provide an image file path instead: {path}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"Photo path is not a regular file: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Unable to read photo file: {path}")

        detections = self._detect(image, min_size=int(min_size))
        if max_faces > 0:
            detections = detections[: int(max_faces)]

        height, width = image.shape[:2]
        source = str(source_path or path)
        created: list[dict[str, Any]] = []
        for x, y, w, h in detections:
            x0, y0, x1, y1 = _expand_box(x, y, w, h, width, height)
            crop = image[y0:y1, x0:x1]
            if crop is None or crop.size == 0:
                continue
            if not self.is_valid_face_crop(crop):
                continue
            arcface_embedding = compute_arcface_embedding(crop)
            embedding = arcface_embedding or compute_simple_embedding(crop)
            quality = estimate_face_quality(crop)
            detector_model = (
                CURRENT_FACE_DETECTOR_MODEL
                if self._insightface is not None
                else FALLBACK_FACE_DETECTOR_MODEL
            )
            embedding_model = (
                CURRENT_FACE_EMBEDDING_MODEL
                if arcface_embedding is not None
                else FALLBACK_FACE_EMBEDDING_MODEL
            )
            face = self.store.add_face(
                embedding=embedding,
                source_type="photo",
                source_path=source,
                timestamp="",
                bbox=[int(x), int(y), int(w), int(h)],
                quality=quality,
                metadata={
                    "ingest": "photo",
                    "detector_model": detector_model,
                    "embedding_model": embedding_model,
                },
            )
            crop_rel = self._save_crop(str(face.get("face_id")), crop)
            face = self.store.update_face(str(face.get("face_id")), crop_path=crop_rel)
            created.append(face)
        return created

    def ingest_vhs(
        self,
        *,
        video_path: str | Path,
        source_path: str | None = None,
        sample_every_seconds: float = 2.0,
        min_size: int = 40,
        max_faces: int = 120,
        max_duration_seconds: float = 0.0,
    ) -> dict[str, Any]:
        self._ensure_primary_model_ready()
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file does not exist: {path}")
        if path.is_dir():
            raise IsADirectoryError(
                f"Video path is a directory. Provide a video file path instead: {path}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"Video path is not a regular file: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {path}")
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 1e-6:
                fps = 29.97
            stride = max(1, int(round(max(0.1, float(sample_every_seconds)) * fps)))
            limit_seconds = float(max_duration_seconds or 0.0)
            source = str(source_path or path)

            frame_idx = 0
            sampled_frames = 0
            created: list[dict[str, Any]] = []
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame_idx % stride != 0:
                    frame_idx += 1
                    continue
                seconds = float(frame_idx) / float(fps)
                if limit_seconds > 0.0 and seconds > limit_seconds:
                    break
                sampled_frames += 1
                boxes = self._detect(frame, min_size=int(min_size))
                height, width = frame.shape[:2]
                for x, y, w, h in boxes:
                    if max_faces > 0 and len(created) >= int(max_faces):
                        break
                    x0, y0, x1, y1 = _expand_box(x, y, w, h, width, height)
                    crop = frame[y0:y1, x0:x1]
                    if crop is None or crop.size == 0:
                        continue
                    if not self.is_valid_face_crop(crop):
                        continue
                    arcface_embedding = compute_arcface_embedding(crop)
                    embedding = arcface_embedding or compute_simple_embedding(crop)
                    quality = estimate_face_quality(crop)
                    timestamp = _timestamp_from_seconds(seconds)
                    detector_model = (
                        CURRENT_FACE_DETECTOR_MODEL
                        if self._insightface is not None
                        else FALLBACK_FACE_DETECTOR_MODEL
                    )
                    embedding_model = (
                        CURRENT_FACE_EMBEDDING_MODEL
                        if arcface_embedding is not None
                        else FALLBACK_FACE_EMBEDDING_MODEL
                    )
                    face = self.store.add_face(
                        embedding=embedding,
                        source_type="vhs",
                        source_path=source,
                        timestamp=timestamp,
                        bbox=[int(x), int(y), int(w), int(h)],
                        quality=quality,
                        metadata={
                            "ingest": "vhs",
                            "frame_index": int(frame_idx),
                            "fps": round(float(fps), 5),
                            "detector_model": detector_model,
                            "embedding_model": embedding_model,
                        },
                    )
                    crop_rel = self._save_crop(str(face.get("face_id")), crop)
                    face = self.store.update_face(
                        str(face.get("face_id")), crop_path=crop_rel
                    )
                    created.append(face)
                if max_faces > 0 and len(created) >= int(max_faces):
                    break
                frame_idx += 1
        finally:
            cap.release()

        return {
            "faces": created,
            "sampled_frames": int(sampled_frames),
            "faces_created": int(len(created)),
            "source_path": str(source_path or path),
        }

    def iter_photo_files(
        self,
        *,
        photo_albums_root: str | Path,
        view_glob: str = "*_View",
        recursive: bool = True,
        extensions: tuple[str, ...] = (".jpg", ".jpeg"),
    ) -> list[Path]:
        root = Path(photo_albums_root)
        if not root.exists():
            raise FileNotFoundError(f"Photo albums root does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Photo albums root is not a directory: {root}")

        ext_set = {str(ext).strip().lower() for ext in extensions if str(ext).strip()}
        if not ext_set:
            ext_set = {".jpg", ".jpeg"}

        files: list[Path] = []
        for view_dir in sorted(root.glob(str(view_glob or "*_View"))):
            if not view_dir.is_dir():
                continue
            iterator = view_dir.rglob("*") if recursive else view_dir.glob("*")
            for path in iterator:
                if not path.is_file():
                    continue
                if path.suffix.lower() not in ext_set:
                    continue
                files.append(path.resolve())
        return files

    def ingest_photo_album_views(
        self,
        *,
        photo_albums_root: str | Path,
        view_glob: str = "*_View",
        recursive: bool = True,
        extensions: tuple[str, ...] = (".jpg", ".jpeg"),
        min_size: int = 40,
        max_faces_per_photo: int = 50,
        max_files: int = 0,
    ) -> dict[str, Any]:
        self._ensure_primary_model_ready()
        photo_files = self.iter_photo_files(
            photo_albums_root=photo_albums_root,
            view_glob=view_glob,
            recursive=recursive,
            extensions=extensions,
        )
        if max_files and max_files > 0:
            photo_files = photo_files[: int(max_files)]

        all_faces: list[dict[str, Any]] = []
        per_photo: list[dict[str, Any]] = []
        for photo_path in photo_files:
            created = self.ingest_photo(
                image_path=photo_path,
                source_path=str(photo_path),
                min_size=min_size,
                max_faces=max_faces_per_photo,
            )
            all_faces.extend(created)
            per_photo.append(
                {
                    "photo_path": str(photo_path),
                    "faces_created": int(len(created)),
                }
            )

        return {
            "photo_files_scanned": int(len(photo_files)),
            "faces_created": int(len(all_faces)),
            "faces": all_faces,
            "per_photo": per_photo,
            "photo_albums_root": str(Path(photo_albums_root)),
            "view_glob": str(view_glob),
        }
