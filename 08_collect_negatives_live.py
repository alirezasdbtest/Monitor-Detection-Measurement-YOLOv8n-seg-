"""جمع‌آوری hard negative از RTSP: ذخیره فریم فعلی با کلید N.

  python 08_collect_negatives_live.py
  python 08_collect_negatives_live.py --source rtsp://...

کلیدها: N = ذخیره در negatives | Q = خروج
"""
import argparse
import os
import time
from datetime import datetime

import cv2

import config
from common_cctv import ensure_cctv_dirs

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=config.DEFAULT_RTSP)
    a = ap.parse_args()

    ensure_cctv_dirs()
    out = config.CCTV_NEGATIVES

    cap = cv2.VideoCapture(a.source, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        raise SystemExit(f"RTSP باز نشد: {a.source}")

    print("N = save negative | Q = quit")
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue

        show = frame.copy()
        cv2.putText(
            show,
            f"negatives saved: {saved} | N=save Q=quit",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Collect hard negatives", show)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("n"), ord("N")):
            name = f"neg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            path = out / name
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
            print("saved", path.name)

    cap.release()
    cv2.destroyAllWindows()
    print(f"total negatives: {saved} -> {out}")


if __name__ == "__main__":
    main()
