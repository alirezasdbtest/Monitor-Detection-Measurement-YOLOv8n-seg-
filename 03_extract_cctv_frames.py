"""استخراج فریم متنوع از ویدیو/RTSP برای برچسب‌گذاری (حذف نزدیک‌تکراری‌ها).

  cctv/raw_videos/*.mp4
  cctv/raw_videos/urls.txt   (هر خط یک RTSP؛ # = کامنت)

  python 03_extract_cctv_frames.py
  python 03_extract_cctv_frames.py --source rtsp://... --every 150 --max 300
  python 03_extract_cctv_frames.py --thr 4      # اگر فریم‌ها خیلی کم شد
"""
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def small_gray(img: np.ndarray) -> np.ndarray:
    s = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)


def is_new(small: np.ndarray, last: np.ndarray | None, thr: float) -> bool:
    if last is None:
        return True
    return float(np.mean(cv2.absdiff(small, last))) > thr


def open_capture(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def extract_from_capture(
    cap: cv2.VideoCapture,
    out: Path,
    prefix: str,
    every: int,
    max_frames: int,
    thr: float,
    max_seconds: float = 0,
) -> int:
    saved = 0
    idx = 0
    last: np.ndarray | None = None  # آخرین فریم ذخیره‌شده (برای همین منبع)
    t0 = time.time()
    while cap.isOpened() and saved < max_frames:
        if max_seconds and time.time() - t0 > max_seconds:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            s = small_gray(frame)
            if is_new(s, last, thr):
                last = s
                name = f"{prefix}_{idx:06d}.jpg"
                cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1
        idx += 1
    return saved


def sources(args) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    stamp = datetime.now().strftime("%m%d%H%M")  # هر اجرا گروه جدا (برای split زمانی)
    if args.source:
        items.append((f"stream{stamp}", args.source))
    elif config.DEFAULT_RTSP:
        items.append((f"camera{stamp}", config.DEFAULT_RTSP))

    urls = config.CCTV_VIDEOS / "urls.txt"
    if urls.exists():
        for i, line in enumerate(urls.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if line and not line.startswith("#"):
                items.append((f"url{i:02d}", line))

    for p in sorted(config.CCTV_VIDEOS.glob("*")):
        if p.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
            items.append((p.stem, str(p)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="مسیر ویدیو یا RTSP")
    ap.add_argument("--every", type=int, default=config.EXTRACT_EVERY, help="هر N فریم یک نمونه")
    ap.add_argument("--max", type=int, default=config.EXTRACT_MAX, help="حداکثر فریم از هر منبع")
    ap.add_argument("--thr", type=float, default=config.EXTRACT_DIFF_THR,
                    help="آستانه تفاوت با آخرین فریم ذخیره‌شده؛ کمتر = فریم بیشتر")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="حداکثر زمان خواندن هر منبع (برای RTSP مفید؛ 0 = نامحدود)")
    a = ap.parse_args()

    out = config.CCTV_FRAMES
    out.mkdir(parents=True, exist_ok=True)
    config.CCTV_VIDEOS.mkdir(parents=True, exist_ok=True)

    srcs = sources(a)
    if not srcs:
        raise SystemExit(f"منبعی نیست. urls.txt یا ویدیو در {config.CCTV_VIDEOS}")

    total = 0
    for prefix, path in srcs:
        cap = open_capture(path)
        if not cap.isOpened():
            print(f"SKIP (باز نشد): {path}")
            continue
        n = extract_from_capture(cap, out, prefix, a.every, a.max, a.thr, a.max_seconds)
        cap.release()
        print(f"{prefix}: {n} frames -> {out}")
        total += n

    print(f"\ntotal: {total} frames")
    print("tip: هدف ۲۰۰-۵۰۰ فریم متنوع (روز/شب، آدم در صحنه، مانیتور روشن/خاموش)")
    print("next: python 07_prefill_labels.py  (pre-label + manual fix)")


if __name__ == "__main__":
    main()