import os
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import argparse
import time
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config
from common_cctv import find_weights


WINDOW = "Live CCTV - Monitor Measurement"


# ============================================================
# CAMERA / SCENE PARAMETERS
# ============================================================

IMG_W = 1920
IMG_H = 1080

CAMERA_HEIGHT_CM = 280.0
DISTANCE_CM = 308.86

# لنز ≈ ۲ میلی‌متر
FOV_H_DEG = 95.0
FOV_V_DEG = FOV_H_DEG * (IMG_H / IMG_W)

# ضرایب اصلاح جداگانه (بر اساس آخرین مشاهده)
SCALE_CORRECTION_W = 1.24      # 
SCALE_CORRECTION_H = 1.038     #   (36/43 * 1.24)


# ============================================================
# CALIBRATION
# ============================================================

class CameraCalibration:
    def __init__(self):
        self.enabled = False
        self.K = None
        self.dist = None
        self.new_K = None
        self.map1 = None
        self.map2 = None
        self.image_size = (IMG_W, IMG_H)

    def load(self, path):
        path = Path(path)
        if not path.exists():
            print(f"[Calibration] File not found: {path}")
            return False
        try:
            data = np.load(str(path))
            if "K" not in data or "dist" not in data:
                raise ValueError("Calibration file must contain K and dist")
            self.K = data["K"].astype(np.float64)
            self.dist = data["dist"].astype(np.float64)
            if "image_size" in data:
                size = data["image_size"]
                self.image_size = (int(size[0]), int(size[1]))
            w, h = self.image_size
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(
                self.K, self.dist, (w, h), 1.0, (w, h)
            )
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, self.dist, None, self.new_K, (w, h), cv2.CV_32FC1
            )
            self.enabled = True
            print("[Calibration] Loaded successfully")
            return True
        except Exception as e:
            print(f"[Calibration] Failed: {e}")
            self.enabled = False
            return False

    def undistort(self, frame):
        if not self.enabled:
            return frame
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)


# ============================================================
# MASK CLEANING
# ============================================================

def clean_polygon(polygon, min_points=8):
    if polygon is None:
        return None
    pts = np.asarray(polygon, dtype=np.float32)
    if len(pts) < min_points:
        return None
    pts_int = np.round(pts).astype(np.int32)
    pts_int = np.unique(pts_int, axis=0)
    if len(pts_int) < 3:
        return None
    hull = cv2.convexHull(pts_int.astype(np.float32)).reshape(-1, 2)
    if len(hull) < 3:
        return None
    return hull


def polygon_area(points):
    return abs(cv2.contourArea(np.asarray(points, dtype=np.float32)))


# ============================================================
# PERSPECTIVE MEASUREMENT + SEPARATE SCALE CORRECTION
# ============================================================

def measure_polygon_perspective(
    polygon,
    img_w=IMG_W,
    img_h=IMG_H,
    D_cm=DISTANCE_CM,
    fov_h_deg=FOV_H_DEG,
    fov_v_deg=FOV_V_DEG,
    scale_w=SCALE_CORRECTION_W,
    scale_h=SCALE_CORRECTION_H
):
    """
    اندازه‌گیری با minAreaRect + ضریب جداگانه برای عرض و ارتفاع
    """
    pts = np.asarray(polygon, dtype=np.float32)
    if len(pts) < 4:
        return None, None

    x_norm = (pts[:, 0] - img_w / 2.0) / (img_w / 2.0)
    y_norm = (pts[:, 1] - img_h / 2.0) / (img_h / 2.0)

    fov_h_rad = math.radians(fov_h_deg)
    fov_v_rad = math.radians(fov_v_deg)

    X = D_cm * np.tan(x_norm * (fov_h_rad / 2.0))
    Y = D_cm * np.tan(y_norm * (fov_v_rad / 2.0))

    points_2d = np.stack([X, Y], axis=1).astype(np.float32)

    rect = cv2.minAreaRect(points_2d)
    (center, (w, h), angle) = rect

    # عرض بزرگ‌تر، ارتفاع کوچک‌تر + ضرایب جداگانه
    width_cm  = max(w, h) * scale_w
    height_cm = min(w, h) * scale_h

    return width_cm, height_cm


# ============================================================
# MEASUREMENT SMOOTHING
# ============================================================

class MeasurementSmoother:
    def __init__(self, alpha=0.18):
        self.alpha = alpha
        self.values = {}

    def update(self, track_id, width, height):
        if track_id is None:
            return width, height
        if track_id not in self.values:
            self.values[track_id] = (width, height)
            return width, height
        old_w, old_h = self.values[track_id]
        new_w = self.alpha * width + (1.0 - self.alpha) * old_w
        new_h = self.alpha * height + (1.0 - self.alpha) * old_h
        self.values[track_id] = (new_w, new_h)
        return new_w, new_h

    def remove_old(self, active_ids):
        active_ids = set(x for x in active_ids if x is not None)
        for x in list(self.values.keys()):
            if x not in active_ids:
                del self.values[x]


# ============================================================
# LABEL DRAWING
# ============================================================

def draw_label(frame, anchor, lines, color=(0, 140, 255)):
    x, y = int(anchor[0]), int(anchor[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 2
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    max_width = max(s[0] for s in sizes) + 14
    total_height = sum(s[1] for s in sizes) + 6 * len(lines) + 10

    frame_h, frame_w = frame.shape[:2]
    x = max(0, min(x, frame_w - max_width - 2))
    top = y - total_height - 6
    if top < 0:
        top = y + 6
    if top + total_height > frame_h:
        top = max(0, frame_h - total_height - 2)

    cv2.rectangle(frame, (x, top), (x + max_width, top + total_height), color, -1)
    cv2.rectangle(frame, (x, top), (x + max_width, top + total_height), (255, 255, 255), 1)

    yy = top + 17
    for line, size in zip(lines, sizes):
        cv2.putText(frame, line, (x + 7, yy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        yy += size[1] + 6


# ============================================================
# MAIN STREAM
# ============================================================

def run_stream(model, a, device, calibration, smoother) -> bool:
    kw = dict(
        source=a.source,
        stream=True,
        conf=a.conf,
        iou=a.iou,
        imgsz=a.imgsz,
        device=device,
        verbose=False,
    )

    if a.no_track:
        gen = model.predict(**kw)
    else:
        gen = model.track(persist=True, tracker=a.tracker, **kw)

    for result in gen:
        raw_frame = result.orig_img.copy()
        frame = calibration.undistort(raw_frame) if calibration.enabled else raw_frame

        if result.masks is not None:
            frame = result.plot(boxes=False, masks=True, conf=False, labels=False)

            polygons = result.masks.xy
            active_track_ids = []

            for i, polygon in enumerate(polygons):
                polygon = clean_polygon(polygon)
                if polygon is None:
                    continue
                if polygon_area(polygon) < a.min_area:
                    continue

                width_cm, height_cm = measure_polygon_perspective(
                    polygon,
                    img_w=frame.shape[1],
                    img_h=frame.shape[0]
                )
                if width_cm is None:
                    continue

                conf = cls_id = track_id = None
                if result.boxes is not None and i < len(result.boxes):
                    box = result.boxes[i]
                    conf = float(box.conf[0].cpu())
                    cls_id = int(box.cls[0].cpu())
                    if box.id is not None:
                        track_id = int(box.id[0].cpu())

                if track_id is not None:
                    active_track_ids.append(track_id)

                smooth_w, smooth_h = smoother.update(track_id, width_cm, height_cm)

                name = result.names.get(cls_id, "monitor") if cls_id is not None else "monitor"
                center = np.mean(polygon, axis=0)
                top_y = int(np.min(polygon[:, 1]))

                line1 = f"ID:{track_id} {name}" if track_id is not None else name
                if conf is not None:
                    line1 += f" {conf:.2f}"

                line2 = f"{smooth_w:.1f} x {smooth_h:.1f} cm"
                line3 = f"D={DISTANCE_CM:.1f} cm"

                draw_label(frame, (int(center[0]), top_y), [line1, line2, line3], color=(0, 140, 255))

            smoother.remove_old(active_track_ids)

        else:
            frame = result.orig_img.copy()
            if result.boxes is not None and len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(result.boxes.conf[i])
                    cls_id = int(result.boxes.cls[i])
                    track_id = int(result.boxes.id[i]) if result.boxes.id is not None else None
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    name = result.names.get(cls_id, str(cls_id))
                    label = f"ID:{track_id} {name} {conf:.2f}" if track_id else f"{name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, max(25, y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        count = len(result.masks) if result.masks is not None else (
            len(result.boxes) if result.boxes is not None else 0
        )

        cv2.putText(frame, f"Monitors: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        calib_text = "Lens: UNDISTORTED" if calibration.enabled else "Lens: NOT CALIBRATED"
        cv2.putText(frame, calib_text, (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame,
                    f"D={DISTANCE_CM:.2f} | W×{SCALE_CORRECTION_W:.2f} H×{SCALE_CORRECTION_H:.3f} | Target 56×36",
                    (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "Final corrected scale (W/H separate)",
                    (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            return True
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return True

    return False


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=config.DEFAULT_RTSP)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=config.PREDICT_CONF)
    parser.add_argument("--iou", type=float, default=config.PREDICT_IOU)
    parser.add_argument("--imgsz", type=int, default=config.FINETUNE_IMGSZ)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--no-track", action="store_true")
    parser.add_argument("--retry", type=float, default=3.0)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--min-area", type=float, default=600.0)
    parser.add_argument("--smooth-alpha", type=float, default=0.18)

    a = parser.parse_args()

    if a.weights is None:
        #ft = config.MODELS / "best_cctv.pt"
        ft = config.MODELS / "best.pt"
        weights = ft if ft.exists() else find_weights(None)
    else:
        weights = find_weights(a.weights)

    device = a.device or ("0" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"Model              : {weights}")
    print(f"Device             : {device}")
    print(f"Fixed Distance D   : {DISTANCE_CM:.2f} cm")
    print(f"Scale Correction W : ×{SCALE_CORRECTION_W:.2f}")
    print(f"Scale Correction H : ×{SCALE_CORRECTION_H:.3f}")
    print(f"Target size        : 56 × 36 cm")
    print("=" * 60)

    model = YOLO(str(weights))
    print(f"Model task: {model.task}")
    if model.task != "segment":
        print("\nWARNING: Use a YOLO segmentation model for true polygons.\n")

    calibration = CameraCalibration()
    if a.calibration:
        calibration.load(a.calibration)
    else:
        print("[Calibration] No calibration file → undistortion disabled.")

    smoother = MeasurementSmoother(alpha=a.smooth_alpha)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    print("Opening live stream...  Q = quit")

    try:
        while True:
            try:
                should_exit = run_stream(model, a, device, calibration, smoother)
                if should_exit:
                    break
                print("Stream ended; reconnecting...")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Stream error: {e}")
                print("Reconnecting...")
            model.predictor = None
            time.sleep(a.retry)
    finally:
        cv2.destroyAllWindows()

    print("Finished.")


if __name__ == "__main__":
    main()