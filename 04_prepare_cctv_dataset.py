"""ساخت دیتاست YOLO-seg برای فاین‌تیون (کلاس monitor) با split زمانی و replay داده اصلی.

  cctv/annotated/images + labels  (YOLO-seg polygon، class=0)
      - فقط تصاویری که فایل label دارند وارد می‌شوند (txt خالی = negative بازبینی‌شده)
  cctv/negatives/*.jpg            (فقط فریم‌های بدون هیچ مانیتور)

  python 04_prepare_cctv_dataset.py
  python 04_prepare_cctv_dataset.py --replay-dir D:\\orig_ds --replay-ratio 0.25
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import config
from common_cctv import ensure_cctv_dirs, write_data_yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
Pair = tuple[Path, "Path | None"]


def valid_seg_line(line: str, n_classes: int) -> bool:
    t = line.split()
    try:
        return len(t) >= 7 and len(t) % 2 == 1 and 0 <= int(t[0]) < n_classes
    except ValueError:
        return False


def valid_det_line(line: str, n_classes: int) -> bool:
    t = line.split()
    try:
        return len(t) == 5 and 0 <= int(t[0]) < n_classes
    except ValueError:
        return False


def is_negative(lbl: Path | None) -> bool:
    return lbl is None or not lbl.read_text(encoding="utf-8").strip()


def read_lines(lbl: Path) -> list[str]:
    return [ln for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]


def collect_annotated(img_dir: Path, lbl_dir: Path, n_classes: int) -> list[Pair]:
    """فقط تصاویر دارای فایل label؛ بقیه بازبینی‌نشده‌اند و نادیده گرفته می‌شوند."""
    if not img_dir.exists():
        return []
    pairs: list[Pair] = []
    skipped = 0
    for img in sorted(img_dir.glob("*")):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            skipped += 1
            continue
        bad = any(
            not (valid_seg_line(ln, n_classes) or valid_det_line(ln, n_classes))
            for ln in read_lines(lbl)
        )
        if bad:
            print(f"WARN bad label (skipped): {lbl.name}")
            continue
        pairs.append((img, lbl))
    if skipped:
        print(f"WARN: {skipped} تصویر در annotated بدون فایل label نادیده گرفته شد "
              "(بازبینی‌نشده). اگر negative‌اند به cctv/negatives ببر.")
    return pairs


def collect_negatives() -> list[Pair]:
    return [
        (p, None)
        for p in sorted(config.CCTV_NEGATIVES.glob("*"))
        if p.suffix.lower() in IMG_EXT
    ]


def group_key(p: Path) -> str:
    """url00_000120 -> url00 | camera10031530_000300 -> camera10031530 | neg_* -> neg"""
    if p.name.startswith("neg_"):
        return "neg"
    return p.stem.rsplit("_", 1)[0]


def temporal_split(pairs: list[Pair], ratio: float, gap: int) -> tuple[list[Pair], list[Pair]]:
    """از انتهای هر گروه (منبع) val می‌گیرد و gap فریم حائل را کنار می‌گذارد."""
    groups: dict[str, list[Pair]] = defaultdict(list)
    for item in pairs:
        groups[group_key(item[0])].append(item)

    train: list[Pair] = []
    val: list[Pair] = []
    for _, items in sorted(groups.items()):
        items.sort(key=lambda x: x[0].name)
        n = len(items)
        n_val = int(n * ratio)
        if n < 5 or n_val == 0:
            train += items
            continue
        g = min(gap, max(0, n - n_val - 1))
        train += items[: n - n_val - g]
        val += items[n - n_val:]

    if not val and len(train) > 1:
        print("WARN: val خالی بود؛ آخرین نمونه به val منتقل شد. داده بیشتر لازم است.")
        val.append(train.pop())
    return train, val


def copy_pair(img: Path, lbl: Path | None, img_out: Path, lbl_out: Path, prefix: str):
    shutil.copy2(img, img_out / f"{prefix}{img.name}")
    dst_lbl = lbl_out / f"{prefix}{img.stem}.txt"
    if lbl is not None and lbl.exists():
        shutil.copy2(lbl, dst_lbl)
    else:
        dst_lbl.write_text("", encoding="utf-8")


def collect_replay(replay_dir: Path, n_classes: int) -> list[Pair]:
    img_root, lbl_root = replay_dir / "images", replay_dir / "labels"
    items: list[Pair] = []
    if not img_root.exists():
        print(f"WARN: {img_root} پیدا نشد؛ replay انجام نشد.")
        return items
    for img in sorted(img_root.rglob("*")):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lbl = lbl_root / img.relative_to(img_root).with_suffix(".txt")
        if not lbl.exists():
            continue
        lines = read_lines(lbl)
        if lines and all(
            valid_seg_line(ln, n_classes) or valid_det_line(ln, n_classes) for ln in lines
        ):
            items.append((img, lbl))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=config.VAL_RATIO)
    ap.add_argument("--gap", type=int, default=config.VAL_GAP,
                    help="فریم حائل بین train و val")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--replay-dir", type=Path, default=config.REPLAY_DIR,
                    help="دیتاست اصلی تک‌کلاسه (images/ و labels/) برای جلوگیری از فراموشی")
    ap.add_argument("--replay-ratio", type=float, default=config.REPLAY_RATIO,
                    help="سهم داده اصلی از train نهایی (۰.۲ تا ۰.۳)")
    a = ap.parse_args()

    ensure_cctv_dirs()
    n_classes = len(config.MONITOR_CLASS_NAMES)

    pairs = collect_annotated(
        config.CCTV_ANNOTATED / "images", config.CCTV_ANNOTATED / "labels", n_classes
    )
    all_pairs = pairs + collect_negatives()

    if not all_pairs:
        raise SystemExit(
            "داده CCTV نیست.\n"
            "1) python 03_extract_cctv_frames.py\n"
            "2) python 07_prefill_labels.py و اصلاح دستی\n"
            "3) فریم‌های بدون مانیتور -> cctv/negatives/\n"
            "4) python 04_prepare_cctv_dataset.py"
        )

    train, val = temporal_split(all_pairs, a.val_ratio, a.gap)

    out = config.CCTV_DS
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)

    stats = {"train": 0, "val": 0, "negatives": 0, "positives": 0, "replay": 0}
    for split, items in (("train", train), ("val", val)):
        for img, lbl in items:
            copy_pair(img, lbl, out / "images" / split, out / "labels" / split, "cctv_")
            stats[split] += 1
            stats["negatives" if is_negative(lbl) else "positives"] += 1

    if a.replay_dir and 0 < a.replay_ratio < 1:
        pool = collect_replay(a.replay_dir, n_classes)
        want = int(len(train) * a.replay_ratio / (1 - a.replay_ratio))
        picked = random.Random(a.seed).sample(pool, min(want, len(pool)))
        for i, (img, lbl) in enumerate(picked):
            copy_pair(img, lbl, out / "images" / "train", out / "labels" / "train", f"orig{i:05d}_")
        stats["replay"] = len(picked)
        if len(picked) < want:
            print(f"WARN: فقط {len(picked)} از {want} نمونه replay موجود بود.")

    write_data_yaml(out)

    print("\n=== CCTV dataset ready ===")
    print("path:", out)
    print("stats:", stats)
    if stats["val"] < 20:
        print("WARN: val کوچک است؛ mAP قابل اعتماد نیست.")
    if stats["negatives"] < 20:
        print("WARN: negative کم است (فریم‌هایی بدون مانیتور یا فقط با اشیای شبیه مانیتور).")
    if stats["positives"] < 50:
        print("WARN: نمونه مثبت کم است؛ recall بعد از فاین‌تیون پایین می‌ماند.")


if __name__ == "__main__":
    main()