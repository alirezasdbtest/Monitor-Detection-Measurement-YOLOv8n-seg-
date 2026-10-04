"""توابع مشترک pipeline فاین‌تیون CCTV."""
from pathlib import Path

import config


def find_weights(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"weights not found: {p}")
        return p
    for p in config.WEIGHT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "هیچ وزن آموزش‌دیده‌ای پیدا نشد. best.pt را در monitor_ft/models/ بگذارید."
    )


def ensure_cctv_dirs():
    for d in (
        config.CCTV_VIDEOS,
        config.CCTV_FRAMES,
        config.CCTV_ANNOTATED / "images",
        config.CCTV_ANNOTATED / "labels",
        config.CCTV_UNREVIEWED,
        config.CCTV_NEGATIVES,
        config.MODELS,
    ):
        d.mkdir(parents=True, exist_ok=True)


def find_monitor_class_id(model) -> int:
    """id کلاس مانیتور در مدل (برای prefill)."""
    if config.MONITOR_SOURCE_CLASS_ID is not None:
        return int(config.MONITOR_SOURCE_CLASS_ID)
    names = model.names  # {id: name}
    hits = [i for i, n in names.items() if str(n).lower() in config.MONITOR_NAME_ALIASES]
    if len(hits) == 1:
        return int(hits[0])
    if len(names) == 1:
        return int(next(iter(names)))
    raise SystemExit(
        f"کلاس مانیتور مشخص نشد. کلاس‌های مدل: {names}\n"
        "config.MONITOR_SOURCE_CLASS_ID را دستی تنظیم کن."
    )


def write_data_yaml(ds: Path | None = None) -> Path:
    ds = ds or config.CCTV_DS
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(config.MONITOR_CLASS_NAMES))
    p = ds / "data.yaml"
    p.write_text(
        f"path: {ds.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"test: images/val\nnames:\n{names}\n",
        encoding="utf-8",
    )
    for c in ds.glob("labels/**/*.cache"):
        c.unlink()
    return p