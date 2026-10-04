"""فاین‌تیون YOLO-seg مانیتور روی دیتاست CCTV.

  python 05_finetune_monitor.py --quick
  python 05_finetune_monitor.py --weights ..\\.myvenv\\best.pt --epochs 40
  python 05_finetune_monitor.py --imgsz 960 --freeze 0 --name monitor_cctv_960
"""
import argparse
import json
import platform
import shutil
import time
import traceback

import config
from common_cctv import find_weights, write_data_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 epoch on 10pct of data")
    ap.add_argument("--epochs", type=int, default=config.FINETUNE_EPOCHS)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--lr0", type=float, default=config.FINETUNE_LR0)
    ap.add_argument("--optimizer", default=config.FINETUNE_OPTIMIZER)
    ap.add_argument("--freeze", type=int, default=config.FINETUNE_FREEZE,
                    help="-1 خودکار | 0 بدون freeze | N تعداد لایه‌های فریز")
    ap.add_argument("--imgsz", type=int, default=config.FINETUNE_IMGSZ)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="monitor_cctv_finetune")
    a = ap.parse_args()

    if not (config.CCTV_DS / "images/train").exists():
        raise SystemExit("ابتدا: python 04_prepare_cctv_dataset.py")

    yaml_path = write_data_yaml()

    import torch
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    use_gpu = torch.cuda.is_available()
    device = a.device if a.device is not None else ("0" if use_gpu else "cpu")
    on_cpu = device == "cpu"
    batch = a.batch or (4 if on_cpu else 16)

    print("CPU:", platform.processor(), "| CUDA:", use_gpu, "| device:", device)
    try:
        d = check_det_dataset(str(yaml_path))
        n_train = len(list((config.CCTV_DS / "images/train").glob("*")))
        n_val = len(list((config.CCTV_DS / "images/val").glob("*")))
        print(f"DATASET OK | classes: {d['names']} | train={n_train} val={n_val}")
    except Exception:
        traceback.print_exc()
        raise SystemExit("خطا در دیتاست.")

    freeze = a.freeze
    if freeze < 0:
        freeze = 10 if n_train < 300 else 0
    print(f"optimizer={a.optimizer} lr0={a.lr0} freeze={freeze} imgsz={a.imgsz} batch={batch}")

    weights = find_weights(a.weights)
    print("Fine-tune from:", weights)

    model = YOLO(str(weights))
    t0 = time.time()
    model.train(
        data=str(yaml_path),
        epochs=1 if a.quick else a.epochs,
        fraction=0.1 if a.quick else 1.0,
        imgsz=a.imgsz,
        batch=batch,
        device=device,
        workers=0 if on_cpu else 4,
        optimizer=a.optimizer,      # بدون این، optimizer=auto مقدار lr0 را نادیده می‌گیرد
        lr0=a.lr0,
        lrf=0.01,
        warmup_epochs=3,
        warmup_bias_lr=0.0,
        freeze=freeze if freeze > 0 else None,
        patience=15,
        close_mosaic=10,
        seed=config.SEED,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        project=str(config.RUNS),
        name=a.name,
        exist_ok=True,
        plots=True,
    )
    minutes = (time.time() - t0) / 60
    print(f"\nFine-tune wall time: {minutes:.1f} min")

    best = config.RUNS / a.name / "weights" / "best.pt"
    config.MODELS.mkdir(exist_ok=True)
    if best.exists():
        dst = config.MODELS / "best_cctv.pt"
        shutil.copy2(best, dst)
        print("saved ->", dst, "(original best.pt unchanged)")

    info = {
        "weights_in": str(weights),
        "epochs": 1 if a.quick else a.epochs,
        "optimizer": a.optimizer,
        "lr0": a.lr0,
        "freeze": freeze,
        "imgsz": a.imgsz,
        "n_train": n_train,
        "n_val": n_val,
        "minutes": round(minutes, 2),
    }
    (config.RUNS / f"finetune_info_{a.name}.json").write_text(
        json.dumps(info, indent=1), encoding="utf-8"
    )
    print("\nnext: python 09_eval_compare.py")
    print("then: python 06_predict_cctv.py --source rtsp://...")


if __name__ == "__main__":
    main()