# Monitor Detection & Measurement (YOLOv8n-seg)

Detects computer monitors in camera images with an instance-segmentation model, then converts each monitor's mask into an estimated physical width and height in centimeters. It works on RTSP streams, video files and images, and tracks each monitor over time.

The system has two parts:

| Part | Role |
|---|---|
| **YOLOv8n-seg model** | Finds each monitor and returns its polygon mask ("the eyes") |
| **Measurement engine** (`06_predict_cctv.py`) | Converts the polygon from pixels to centimeters ("the ruler") |

Final accuracy depends on both parts, and also on camera calibration, camera-to-monitor distance, monitor angle and mask quality.

> Fine-tuning on CCTV footage (frame extraction, labeling in CVAT, training, evaluation) is documented separately and is not covered here.

---

## Project layout

```
monitor_ft/
├── config.py             # paths, default RTSP, thresholds
├── common_cctv.py        # weight lookup and shared helpers
├── 06_predict_cctv.py    # live detection + measurement
├── models/
│   ├── best.pt           # base single-class (Monitor) segmentation model
│   └── best_cctv.pt      # optional: model adapted to the camera (used first if present)
└── Saidi1.ipynb          # dataset exploration, geometry analysis, base training (Colab)
```

## Requirements

```bash
pip install ultralytics opencv-python numpy torch psutil
```

`psutil` is optional (only used to show CPU/RAM on screen). A GPU is not required, but the live program runs at a few FPS on CPU only.

## Quick start

1. Put the base weights at `models/best.pt`.
2. Set the stream in `config.py` (`DEFAULT_RTSP`) or pass it on the command line.
3. Run:

```bash
python 06_predict_cctv.py --source rtsp://<user>:<password>@<ip>:554
python 06_predict_cctv.py --source path/to/video.mp4
```

Press **Q** to quit. The window shows each mask with `ID`, confidence and `width x height cm`.

**Security note:** do not commit camera credentials. Keep `DEFAULT_RTSP` out of version control (for example, read it from an environment variable).

---

## 1. Dataset (base model)

The base model was trained on **LabEquipVis**, a computer-lab equipment dataset on Kaggle (`bmshahriaalam/labequipvis-dataset-of-computer-lab-equipment`), captured with an Oppo Reno8 Pro at East West University. Exported images are 640×640, and annotations are polygons (about 91.7% are valid polygons). Only the **Monitor** class is used.

| Split | Images | Labels |
|---|---|---|
| train | 1809 | 1809 |
| valid | 516 | 516 |
| test | 259 | 259 |

After keeping only Monitor: **2176 images, 19,074 monitor instances** (train 1526, valid 432, test 218), with a single-class `data.yaml` (`0: Monitor`).

Other classes in the original dataset include Chair, Mouse, CPU, Keyboard, Light, Fire Extinguisher, Projector, Digital board and AC.

## 2. Geometry analysis on the dataset

Before building the measurement engine, the dataset's polygons were studied to see what can be measured reliably.

1. **BBox vs. MinAreaRect vs. PCA.** In perspective images these rectangles do not always match the screen's four real corners, because they depend on the distribution of polygon points.
2. **Four-corner extraction.** Polygons were approximated to quadrilaterals (`approxPolyDP`, epsilon 0.015). Samples were kept only if they are at least 5 px from the image border, have at least 6 points, form a valid quadrilateral and are not too small. 6,550 of 10,203 polygons (64%) were fully in view.
3. **Stand removal.** Some annotations include the stand; only "screen only" polygons are useful. Morphological cleaning with an adaptive kernel replaced a fixed 17×17 one.
4. **Quality control.** A geometric check on coverage/compactness left **2,737 good** quadrilaterals (178 rejected).
5. **Aspect-ratio clustering.** Good samples were grouped by aspect ratio against standard ratios (16:9, 16:10, 5:4, 4:3) and assumed standard diagonals to estimate a rough mm-per-pixel scale. 2,079 samples (76%) were considered reliable.

Takeaway: the polygon gives a good shape, but pixel width/height is not physical width/height. Perspective, camera settings and what the polygon includes (stand, overlaps) all change the result. The scale estimate from step 5 is approximate and should be checked against real monitor sizes.

> Some counts in the notebook come from different filtering stages (for example 17,344, 19,074 and 12,193 monitor annotations). Check them against the notebook before citing them.

## 3. Base model

`yolov8n-seg` (3.26 M parameters, 11.3 GFLOPs): the lightest YOLOv8 segmentation model, suitable for CPU and edge devices, with masks that follow tilted monitors better than boxes.

Validation (Colab, Tesla T4, 400 images, 3,391 instances):

| | Box | Mask |
|---|---|---|
| Precision | 0.946 | 0.947 |
| Recall | 0.959 | 0.961 |
| mAP50 | 0.975 | 0.975 |
| mAP50-95 | 0.834 | 0.803 |

Speed: 1.3 ms preprocess, 4.4 ms inference, 5.0 ms postprocess per image (GPU).

---

## 4. Live measurement program (`06_predict_cctv.py`)

### Pipeline

1. Read frames from RTSP or a video file.
2. Run the segmentation model, with ByteTrack tracking by default.
3. Take each monitor's polygon from `result.masks.xy`.
4. Clean the polygon (unique points, convex hull) and discard small masks (`--min-area`).
5. Convert polygon points to centimeters with the perspective model.
6. Fit a minimum-area rotated rectangle and read its two sides.
7. Apply separate empirical correction factors for width and height.
8. Smooth values per track ID and draw the label.

### Perspective model

```
x_n = (x - W/2) / (W/2)         y_n = (y - H/2) / (H/2)
X = D * tan(x_n * θh / 2)       Y = D * tan(y_n * θv / 2)
```

Then `cv2.minAreaRect` is applied on `(X, Y)`; the longer side is multiplied by `SCALE_CORRECTION_W` (1.24) and the shorter by `SCALE_CORRECTION_H` (1.038).

### Camera/scene parameters (top of the file)

| Constant | Value | Meaning |
|---|---|---|
| `IMG_W`, `IMG_H` | 1920, 1080 | Frame size |
| `DISTANCE_CM` | 308.86 | Assumed camera-to-monitor distance |
| `CAMERA_HEIGHT_CM` | 280 | Camera height (informational) |
| `FOV_H_DEG` | 95 | Horizontal field of view (≈2 mm lens) |
| `FOV_V_DEG` | `FOV_H * H/W` | Vertical FOV (linear approximation) |
| `SCALE_CORRECTION_W/H` | 1.24 / 1.038 | Empirical factors, tuned on one 56×36 cm monitor |

**These constants are specific to one camera, one distance and one target monitor. Recalibrate them for any new setup.**

### Smoothing and tracking

ByteTrack gives each monitor a stable ID. Measurements are smoothed per ID with an exponential moving average, `d_hat = α·d + (1-α)·d_hat_prev`, default `α = 0.18`. A smaller α is steadier but reacts more slowly. It reduces noise but cannot remove a constant calibration error.

### Command-line options

| Option | Default | Description |
|---|---|---|
| `--source` | `config.DEFAULT_RTSP` | RTSP URL, video file |
| `--weights` | auto | Uses `models/best_cctv.pt` if present, otherwise `best.pt` |
| `--conf` / `--iou` | 0.45 / 0.45 | Detection thresholds |
| `--imgsz` | 640 | Inference size |
| `--device` | auto | `0` for GPU, `cpu` otherwise |
| `--tracker` | `bytetrack.yaml` | Tracker config |
| `--no-track` | off | Disable tracking (no IDs, no smoothing) |
| `--min-area` | 600 | Drop masks smaller than this (px²) |
| `--smooth-alpha` | 0.18 | EMA factor |
| `--calibration` | none | `.npz` with `K`, `dist` (and optional `image_size`) for lens undistortion |
| `--retry` | 3.0 | Seconds before reconnecting a dropped stream |

The CPU-optimized version of the script also supports `--threads`, `--vid-stride`, `--show-scale` and `--no-stats`.

### Lens calibration

With `--calibration file.npz`, frames and polygon points can be undistorted. Without it the screen shows `Lens: NOT CALIBRATED`. Note that the empirical correction factors above were tuned without calibration, so retune them if you enable it.

### Performance tips (CPU)

- Lower the camera FPS (8–10 is enough); decoding a 25 fps 1080p stream competes with inference.
- Try `--threads` values around the number of physical cores.
- Use `--show-scale 0.5` to shrink the display window without changing measurements.
- Export to OpenVINO for Intel CPUs if it installs on your Python version.

---

## Known limitations

- **Fixed distance.** `DISTANCE_CM` is one value for all monitors, so only monitors at roughly that depth are measured correctly. Monitors nearer or farther will be wrong; they need depth estimation (for example a ground-plane homography).
- **MinAreaRect is not the screen.** If the mask includes the stand, a bezel or neighboring objects, the size is overestimated.
- **Empirical factors.** The 1.24/1.038 factors come from a single monitor and may not generalize.
- **False positives.** Large flat surfaces (a window, a curtain, a desk corner) can be detected as monitors and produce unrealistic sizes. Filtering results outside a plausible size range, and adding hard negatives to training, both help.
- **Back-facing and edge-clipped monitors** give unreliable measurements.
- **Frame rate.** On CPU only, expect a few FPS.

## Suggested next steps

1. Validate measured sizes against several monitors with known dimensions.
2. Replace the fixed distance with a per-monitor depth estimate.
3. Apply real lens calibration and retune the correction factors.
4. Add a plausible-size filter to hide unrealistic outputs.
