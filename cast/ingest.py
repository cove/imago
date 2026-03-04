from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .storage import TextFaceStore

YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


def _timestamp_from_seconds(seconds: float) -> str:
    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    whole_seconds, millis = divmod(rem, 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(whole_seconds):02d}.{int(millis):03d}"


def _expand_box(x: int, y: int, w: int, h: int, width: int, height: int, margin: float = 0.22) -> tuple[int, int, int, int]:
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
    resized = cv2.resize(gray, (int(out_size), int(out_size)), interpolation=cv2.INTER_AREA)
    vec = resized.astype(np.float32).reshape(-1) / 255.0
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec = vec / norm
    return [float(item) for item in vec.tolist()]


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
    min_std: float = 6.0,
    min_dynamic_range: float = 18.0,
    min_laplacian_var: float = 4.0,
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


def _cascade_rows(raw: Any) -> np.ndarray:
    if raw is None:
        return np.empty((0, 4), dtype=np.int32)
    rows = np.asarray(raw)
    if rows.size == 0:
        return np.empty((0, 4), dtype=np.int32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return rows


def grayscale_entropy(gray_u8: np.ndarray) -> float:
    if gray_u8 is None or gray_u8.size == 0:
        return 0.0
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).reshape(-1)
    total = float(np.sum(hist))
    if total <= 0.0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def quantized_unique_colors(image_bgr: np.ndarray, *, levels: int = 8) -> int:
    if image_bgr is None or image_bgr.size == 0:
        return 0
    h, w = image_bgr.shape[:2]
    side = max(16, min(40, int(min(h, w))))
    small = cv2.resize(image_bgr, (side, side), interpolation=cv2.INTER_AREA)
    step = max(1, int(256 // max(2, int(levels))))
    quant = (small // step).astype(np.uint8)
    flat = quant.reshape(-1, 3)
    return int(np.unique(flat, axis=0).shape[0])


def _ensure_yunet_model(model_path: Path) -> bool:
    if model_path.exists():
        return True
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(YUNET_MODEL_URL, timeout=20) as response:
            data = response.read()
        model_path.write_bytes(data)
    except Exception:
        return False
    return model_path.exists()


class FaceIngestor:
    def __init__(self, store: TextFaceStore):
        self.store = store
        cascades = Path(cv2.data.haarcascades)
        face_path = cascades / "haarcascade_frontalface_default.xml"
        strict_face_path = cascades / "haarcascade_frontalface_alt2.xml"
        eye_path = cascades / "haarcascade_eye_tree_eyeglasses.xml"
        profile_path = cascades / "haarcascade_profileface.xml"
        upper_body_path = cascades / "haarcascade_upperbody.xml"

        self._cascade = cv2.CascadeClassifier(str(face_path))
        self._strict_face = cv2.CascadeClassifier(str(strict_face_path))
        self._eye = cv2.CascadeClassifier(str(eye_path))
        self._profile = cv2.CascadeClassifier(str(profile_path))
        self._upper_body = cv2.CascadeClassifier(str(upper_body_path))

        model_dir = Path(__file__).resolve().parent / "models"
        model_path = model_dir / "face_detection_yunet_2023mar.onnx"
        self._yunet = None
        self._yunet_score_threshold = 0.86
        self._yunet_verify_threshold = 0.90
        self.skip_artwork_default = True
        if _ensure_yunet_model(model_path):
            try:
                self._yunet = cv2.FaceDetectorYN_create(
                    str(model_path),
                    "",
                    (320, 320),
                    float(self._yunet_score_threshold),
                    0.3,
                    5000,
                )
            except Exception:
                self._yunet = None

        if self._cascade.empty():
            raise RuntimeError(f"Unable to load face cascade: {face_path}")
        if self._strict_face.empty():
            raise RuntimeError(f"Unable to load strict face cascade: {strict_face_path}")
        if self._eye.empty():
            raise RuntimeError(f"Unable to load eye cascade: {eye_path}")
        if self._profile.empty():
            raise RuntimeError(f"Unable to load profile cascade: {profile_path}")
        if self._upper_body.empty():
            raise RuntimeError(f"Unable to load upper-body cascade: {upper_body_path}")

    def looks_like_artwork_face(self, crop_bgr: np.ndarray) -> bool:
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)

        sat = hsv[:, :, 1].astype(np.float32)
        sat_mean = float(np.mean(sat))
        sat_std = float(np.std(sat))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        entropy = float(grayscale_entropy(gray))
        edges = cv2.Canny(gray, 60, 150)
        edge_density = float(np.mean(edges > 0))
        unique_colors = int(quantized_unique_colors(crop_bgr, levels=8))

        cr = ycrcb[:, :, 1].astype(np.float32)
        cb = ycrcb[:, :, 2].astype(np.float32)
        skin_mask = (
            (cr > 133.0)
            & (cr < 173.0)
            & (cb > 77.0)
            & (cb < 127.0)
        )
        skin_ratio = float(np.mean(skin_mask))

        score = 0.0
        if skin_ratio < 0.05:
            score += 0.30
        if sat_mean < 38.0:
            score += 0.18
        if sat_std < 24.0:
            score += 0.12
        if lap_var < 24.0:
            score += 0.16
        if entropy < 4.8:
            score += 0.12
        if edge_density > 0.20:
            score += 0.10
        if edge_density < 0.04 and sat_mean < 34.0:
            score += 0.08
        if unique_colors < 50:
            score += 0.10

        eye_count = self._count_eyes(gray)
        strict_hit = self._has_strict_frontal_face(gray)
        profile_hit = self._has_profile_face(gray)
        if eye_count >= 2 and (strict_hit or profile_hit):
            score -= 0.22
        if skin_ratio > 0.09:
            score -= 0.20
        if lap_var > 36.0 and entropy > 5.2:
            score -= 0.06
        if sat_mean > 45.0 and sat_std > 20.0:
            score -= 0.06
        return bool(score >= 0.68)

    def is_valid_face_crop(
        self,
        crop_bgr: np.ndarray,
        *,
        skip_artwork: bool | None = None,
    ) -> bool:
        if not crop_has_visual_detail(crop_bgr):
            return False
        if not self._passes_face_sanity(crop_bgr):
            return False
        artwork_mode = self.skip_artwork_default if skip_artwork is None else bool(skip_artwork)
        if artwork_mode and self.looks_like_artwork_face(crop_bgr):
            return False
        return True

    def _yunet_detect_rows(
        self,
        image_bgr: np.ndarray,
        *,
        score_threshold: float,
    ) -> np.ndarray:
        detector = self._yunet
        if detector is None or image_bgr is None or image_bgr.size == 0:
            return np.empty((0, 15), dtype=np.float32)
        h, w = image_bgr.shape[:2]
        if h < 12 or w < 12:
            return np.empty((0, 15), dtype=np.float32)
        try:
            detector.setInputSize((int(w), int(h)))
            _ok, faces = detector.detect(image_bgr)
        except cv2.error:
            return np.empty((0, 15), dtype=np.float32)
        rows = np.asarray(faces) if faces is not None else np.empty((0, 15), dtype=np.float32)
        if rows.size == 0:
            return np.empty((0, 15), dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        out_rows = []
        for row in rows:
            if int(len(row)) < 15:
                continue
            score = float(row[14])
            if score < float(score_threshold):
                continue
            out_rows.append(np.asarray(row, dtype=np.float32))
        if not out_rows:
            return np.empty((0, 15), dtype=np.float32)
        return np.vstack(out_rows)

    def _yunet_row_to_box(
        self,
        row: np.ndarray,
        *,
        image_w: int,
        image_h: int,
    ) -> tuple[int, int, int, int] | None:
        if row is None or int(len(row)) < 4:
            return None
        x = int(round(float(row[0])))
        y = int(round(float(row[1])))
        w = int(round(float(row[2])))
        h = int(round(float(row[3])))
        if w <= 0 or h <= 0:
            return None
        x = max(0, min(x, int(image_w) - 1))
        y = max(0, min(y, int(image_h) - 1))
        x2 = max(x + 1, min(x + w, int(image_w)))
        y2 = max(y + 1, min(y + h, int(image_h)))
        ww = int(x2 - x)
        hh = int(y2 - y)
        if ww <= 0 or hh <= 0:
            return None
        return int(x), int(y), int(ww), int(hh)

    def _yunet_landmarks_plausible(
        self,
        row: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> bool:
        if row is None or int(len(row)) < 15:
            return False
        x, y, w, h = [int(v) for v in box]
        if w <= 0 or h <= 0:
            return False
        lx, ly = float(row[4]), float(row[5])
        rx, ry = float(row[6]), float(row[7])
        nx, ny = float(row[8]), float(row[9])
        mlx, mly = float(row[10]), float(row[11])
        mrx, mry = float(row[12]), float(row[13])
        if not (lx < rx and mlx < mrx):
            return False
        eye_y_delta = abs(ly - ry)
        if eye_y_delta > float(h) * 0.22:
            return False
        eye_mid_y = float(ly + ry) * 0.5
        mouth_mid_y = float(mly + mry) * 0.5
        if not (eye_mid_y < ny < mouth_mid_y):
            return False
        if (mouth_mid_y - eye_mid_y) < float(h) * 0.16:
            return False
        eye_distance = abs(rx - lx)
        if eye_distance < float(w) * 0.16 or eye_distance > float(w) * 0.90:
            return False
        margin_x = float(w) * 0.18
        margin_y = float(h) * 0.20
        min_x = float(x) - margin_x
        max_x = float(x + w) + margin_x
        min_y = float(y) - margin_y
        max_y = float(y + h) + margin_y
        for px, py in [(lx, ly), (rx, ry), (nx, ny), (mlx, mly), (mrx, mry)]:
            if px < min_x or px > max_x or py < min_y or py > max_y:
                return False
        return True

    def _detect_yunet(
        self,
        image_bgr: np.ndarray,
        *,
        min_size: int = 40,
        score_threshold: float | None = None,
        require_center: bool = False,
    ) -> list[tuple[int, int, int, int]]:
        h, w = image_bgr.shape[:2]
        rows = self._yunet_detect_rows(
            image_bgr,
            score_threshold=float(
                self._yunet_score_threshold if score_threshold is None else score_threshold
            ),
        )
        if rows.size == 0:
            return []
        out: list[tuple[int, int, int, int]] = []
        for row in rows:
            box = self._yunet_row_to_box(row, image_w=int(w), image_h=int(h))
            if box is None:
                continue
            x, y, ww, hh = box
            if min(ww, hh) < int(max(1, int(min_size))):
                continue
            if not self._yunet_landmarks_plausible(row, box):
                continue
            if require_center:
                area_ratio = float(ww * hh) / float(max(1, w * h))
                if area_ratio < 0.10 or area_ratio > 0.98:
                    continue
                cx = float(x + (ww * 0.5))
                cy = float(y + (hh * 0.5))
                if abs(cx - (float(w) * 0.5)) > float(w) * 0.34:
                    continue
                if abs(cy - (float(h) * 0.48)) > float(h) * 0.36:
                    continue
            out.append((int(x), int(y), int(ww), int(hh)))
        return out

    def _safe_detect(
        self,
        cascade: cv2.CascadeClassifier,
        gray: np.ndarray,
        *,
        scale_factor: float,
        min_neighbors: int,
        min_size: int,
    ) -> np.ndarray:
        if gray is None or gray.size == 0:
            return np.empty((0, 4), dtype=np.int32)
        h, w = gray.shape[:2]
        if h < 2 or w < 2:
            return np.empty((0, 4), dtype=np.int32)
        side = int(max(1, int(min_size)))
        if side > min(h, w):
            return np.empty((0, 4), dtype=np.int32)
        try:
            raw = cascade.detectMultiScale(
                gray,
                scaleFactor=float(scale_factor),
                minNeighbors=int(min_neighbors),
                minSize=(int(side), int(side)),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except cv2.error:
            return np.empty((0, 4), dtype=np.int32)
        return _cascade_rows(raw)

    def _count_eyes(self, gray_face: np.ndarray) -> int:
        h, w = gray_face.shape[:2]
        if h < 20 or w < 20:
            return 0
        top = gray_face[: max(1, int(h * 0.72)), :]
        min_eye = max(8, min(top.shape[0], top.shape[1]) // 8)
        eyes = self._safe_detect(
            self._eye,
            top,
            scale_factor=1.1,
            min_neighbors=4,
            min_size=int(min_eye),
        )
        return int(len(eyes))

    def _has_profile_face(self, gray_face: np.ndarray) -> bool:
        h, w = gray_face.shape[:2]
        if h < 24 or w < 24:
            return False
        min_side = max(18, min(h, w) // 3)
        prof = self._safe_detect(
            self._profile,
            gray_face,
            scale_factor=1.1,
            min_neighbors=4,
            min_size=int(min_side),
        )
        if len(prof) > 0:
            return True
        flipped = cv2.flip(gray_face, 1)
        prof_flip = self._safe_detect(
            self._profile,
            flipped,
            scale_factor=1.1,
            min_neighbors=4,
            min_size=int(min_side),
        )
        return len(prof_flip) > 0

    def _has_strict_frontal_face(self, gray_face: np.ndarray) -> bool:
        h, w = gray_face.shape[:2]
        if h < 28 or w < 28:
            return False
        min_side = max(22, min(h, w) // 3)
        rows = self._safe_detect(
            self._strict_face,
            gray_face,
            scale_factor=1.05,
            min_neighbors=6,
            min_size=int(min_side),
        )
        if rows.size == 0:
            return False
        cx = float(w) * 0.5
        cy = float(h) * 0.45
        for row in rows:
            if int(len(row)) < 4:
                continue
            x, y, ww, hh = [int(v) for v in row[:4]]
            if ww <= 0 or hh <= 0:
                continue
            rcx = float(x + (ww * 0.5))
            rcy = float(y + (hh * 0.5))
            if abs(rcx - cx) <= float(w) * 0.28 and abs(rcy - cy) <= float(h) * 0.32:
                return True
        return False

    def _looks_like_upper_body(self, gray_face: np.ndarray) -> bool:
        h, w = gray_face.shape[:2]
        if h < 28 or w < 28:
            return False
        min_side = max(18, min(h, w) // 3)
        rows = self._safe_detect(
            self._upper_body,
            gray_face,
            scale_factor=1.1,
            min_neighbors=3,
            min_size=int(min_side),
        )
        return len(rows) > 0

    def _passes_face_sanity(self, crop_bgr: np.ndarray) -> bool:
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        h, w = crop_bgr.shape[:2]
        if h < 24 or w < 24:
            return False
        aspect = float(w) / float(max(1, h))
        if aspect < 0.55 or aspect > 1.75:
            return False
        if self._yunet is not None:
            confirmed = self._detect_yunet(
                crop_bgr,
                min_size=max(18, min(int(h), int(w)) // 6),
                score_threshold=float(self._yunet_verify_threshold),
                require_center=True,
            )
            if not confirmed:
                return False

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        eye_count = self._count_eyes(gray)
        profile_hit = self._has_profile_face(gray)
        strict_hit = self._has_strict_frontal_face(gray)
        upper_body_hit = self._looks_like_upper_body(gray)
        if self._yunet is not None:
            # YuNet confirmation already passed; keep this as weak secondary check only.
            if upper_body_hit and not strict_hit:
                return False
            return True
        signal = 0
        if eye_count >= 2:
            signal += 2
        elif eye_count == 1:
            signal += 1
        if profile_hit:
            signal += 1
        if strict_hit:
            signal += 2
        if upper_body_hit:
            signal -= 2
        return signal >= 2

    def _detect(self, image_bgr: np.ndarray, *, min_size: int = 40) -> list[tuple[int, int, int, int]]:
        if self._yunet is not None:
            return self._detect_yunet(image_bgr, min_size=int(min_size))

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rows = self._safe_detect(
            self._cascade,
            gray,
            scale_factor=1.1,
            min_neighbors=5,
            min_size=int(min_size),
        )
        if rows.size == 0:
            return []

        out = []
        for row in rows:
            if int(len(row)) < 4:
                continue
            x, y, w, h = [int(v) for v in row]
            if w <= 0 or h <= 0:
                continue
            out.append((x, y, w, h))
        return out

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
        skip_artwork: bool | None = None,
    ) -> list[dict[str, Any]]:
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
            if not self.is_valid_face_crop(crop, skip_artwork=skip_artwork):
                continue
            embedding = compute_simple_embedding(crop)
            quality = estimate_face_quality(crop)
            face = self.store.add_face(
                embedding=embedding,
                source_type="photo",
                source_path=source,
                timestamp="",
                bbox=[int(x), int(y), int(w), int(h)],
                quality=quality,
                metadata={"ingest": "photo"},
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
        skip_artwork: bool | None = None,
    ) -> dict[str, Any]:
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
                    if not self.is_valid_face_crop(crop, skip_artwork=skip_artwork):
                        continue
                    embedding = compute_simple_embedding(crop)
                    quality = estimate_face_quality(crop)
                    timestamp = _timestamp_from_seconds(seconds)
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
                        },
                    )
                    crop_rel = self._save_crop(str(face.get("face_id")), crop)
                    face = self.store.update_face(str(face.get("face_id")), crop_path=crop_rel)
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
        skip_artwork: bool | None = None,
    ) -> dict[str, Any]:
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
                skip_artwork=skip_artwork,
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
