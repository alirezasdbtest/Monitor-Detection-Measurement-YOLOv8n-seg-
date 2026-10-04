"""پیش‌برچسب‌گذاری فریم‌های CCTV با مدل فعلی (برای اصلاح سریع‌تر).

- فریم‌هایی که مدل مانیتور دیده -> cctv/annotated/images + labels (class همیشه 0)
- فریم‌هایی که مدل چیزی ندیده -> cctv/unreviewed  (دستی ببین: مانیتور دارد؟ برچسب بزن
  و به annotated ببر | ندارد؟ به cctv/negatives ببر)
- برچسب‌های دستی/قبلاً پردازش‌شده هرگز بازنویسی یا حذف نمی‌شوند.

  python 07_prefill_labels.py
  python 07_prefill_labels.py --conf 0.35 --max 200
"""
import argparse
import shutil
from pathlib import Path

import config
from common_cctv import ensure_cctv_dirs, find_monitor_class_id, find_weights

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def remap_class_to_zero(src: Path, dst: Path) -> int:
    """کپی label با تبدیل id کلاس به 0؛ تعداد خطوط را برمی‌گرداند."""
    lines = []
    for ln in src.read_text(encoding="utf-8").splitlines():
        t = ln.split()
        if len(t) >= 5:
            lines.append(" ".join(["0"] + t[1:]))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=config.CCTV_FRAMES)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--max", type=int, default=0, help="0 = همه فریم‌ها")
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()

    ensure_cctv_dirs()
    img_out = config.CCTV_ANNOTATED / "images"
    lbl_out = config.CCTV_ANNOTATED / "labels"

    all_frames = sorted(p for p in a.frames.glob("*") if p.suffix.lower() in IMG_EXT)
    if not all_frames:
        raise SystemExit(f"فریمی نیست. ابتدا 03_extract_cctv_frames.py -> {a.frames}")

    # فریم‌هایی که قبلاً جایی رفته‌اند دوباره پردازش نمی‌شوند
    done = (
        {p.name for p in img_out.glob("*")}
        | {p.name for p in config.CCTV_UNREVIEWED.glob("*")}
        | {p.name for p in config.CCTV_NEGATIVES.glob("*")}
    )
    frames = [p for p in all_frames if p.name not in done]
    if a.max > 0:
        frames = frames[: a.max]
    if not frames:
        raise SystemExit("همه فریم‌ها قبلاً پردازش شده‌اند (چیز جدیدی نیست).")

    weights = find_weights(a.weights)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    cls_id = find_monitor_class_id(model)
    print(f"model classes: {model.names} | monitor id = {cls_id}")

    tmp = config.OUTPUTS / "_prefill_run"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    print(f"Prefill {len(frames)} frames with {weights} conf={a.conf}")
    for _ in model.predict(
        source=[str(p) for p in frames],
        conf=a.conf,
        imgsz=config.FINETUNE_IMGSZ,
        classes=[cls_id],
        stream=True,
        save=False,
        save_txt=True,
        project=str(tmp),
        name="labels",
        exist_ok=True,
        verbose=False,
    ):
        pass

    lbl_dir = tmp / "labels" / "labels"
    if not lbl_dir.exists():
        lbl_dir = tmp / "labels"
    if not lbl_dir.exists():
        # هیچ تشخیصی روی هیچ فریمی -> همه به unreviewed
        lbl_dir.mkdir(parents=True, exist_ok=True)

    n_lbl = n_unrev = 0
    for img in frames:
        dst_lbl = lbl_out / f"{img.stem}.txt"
        if dst_lbl.exists():  # برچسب دستی؛ دست نزن
            continue
        src_lbl = lbl_dir / f"{img.stem}.txt"
        if src_lbl.exists() and remap_class_to_zero(src_lbl, dst_lbl) > 0:
            shutil.copy2(img, img_out / img.name)
            n_lbl += 1
        else:
            if dst_lbl.exists():
                dst_lbl.unlink()
            shutil.copy2(img, config.CCTV_UNREVIEWED / img.name)
            n_unrev += 1

    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nwith labels -> {img_out}: {n_lbl}")
    print(f"no detection -> {config.CCTV_UNREVIEWED}: {n_unrev}")
    print(
        "\nManual steps:\n"
        "  1) annotated: mask اشتباه (صندلی، کیس) را حذف و مانیتور جاافتاده را اضافه کن\n"
        "  2) unreviewed: مانیتور دارد -> برچسب بزن و (عکس+label) را به annotated ببر\n"
        "                 مانیتور ندارد -> عکس را به cctv/negatives ببر\n"
        "  3) python 04_prepare_cctv_dataset.py\n"
        "توجه: عکس بدون فایل label داخل annotated نادیده گرفته می‌شود "
        "(فایل .txt خالی = بازبینی‌شده و negative)."
    )


if __name__ == "__main__":
    main()