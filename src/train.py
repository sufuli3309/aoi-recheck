"""训练：YOLOv8-cls 基线的二分类复判模型。

为什么是分类模型而不是检测模型？
    真实项目里模型吃到的是 AOI 已经裁好的缺陷 crop，缺陷位置由 AOI 给出，
    再上一层检测属于重复劳动。这里用分类头，算力与延迟都更划算。
    （若 AOI 只给整图、位置未知，则应换检测头，见 docs/decisions.md）

为什么不做类别重采样？
    重采样会改动先验，让模型输出的概率不再对应当前产线真实分布，
    后续阈值就失去了业务含义。不平衡交给**推理期的代价敏感阈值**处理，
    这样"偏保守还是偏激进"是一个可调、可解释、可向客户交代的旋钮。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    m, d = cfg["model"], cfg["data"]
    data_root = Path(d["root"])

    for split in ("train", "val"):
        if not (data_root / split).is_dir():
            raise SystemExit(f"缺少 {data_root/split}，先跑 data/build_dataset.py")

    weights = m["weights"]
    if weights is None:
        # 从零开始：用 ultralytics 的 yaml 定义（无预训练权重）
        model = YOLO("yolov8n-cls.yaml")
    else:
        model = YOLO(weights)

    model.train(
        data=str(data_root),
        epochs=args.epochs or m["epochs"],
        imgsz=d["imgsz"],
        batch=m["batch"],
        lr0=m["lr0"],
        patience=m["patience"],
        device=m["device"],
        workers=d["workers"],
        seed=cfg.get("seed", 42),
        project=m["project"],
        name=m["name"],
        exist_ok=True,
        pretrained=weights is not None,
        plots=True,
        verbose=True,
    )
    # ultralytics 会在 project 下再套一层任务目录，实际路径以 trainer 为准，
    # 写下来给 evaluate.py / serve.py 用，避免两边路径对不上
    save_dir = Path(model.trainer.save_dir)
    ckpt = save_dir / "weights" / "best.pt"
    Path("runs").mkdir(exist_ok=True)
    Path("runs/last_ckpt.txt").write_text(str(ckpt.resolve()), encoding="utf-8")
    print(f"[train] 完成，最佳权重：{ckpt}")


if __name__ == "__main__":
    main()
