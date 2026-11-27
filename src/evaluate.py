"""评估：不只看 accuracy。

产线关心的从来不是"准确率多少"，而是两个数：
    1. 漏检率（真缺陷被判成伪）—— 缺陷件流出，最贵的错误
    2. 过杀率（伪缺陷被判成真）—— 白派一次人工复核，便宜

所以这里做三件事：
    a. 给出 AUC / AP / 混淆矩阵（对外汇报用）
    b. 扫描阈值，画出「期望代价 vs 阈值」，取代价最小点
    c. 把最优阈值及其对应的漏检/过杀一起写进 metrics.json（部署用）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRUE_NG = "true_ng"


def collect(model: YOLO, test_dir: Path, imgsz: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    files = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    y_true, y_score, names = [], [], []
    for p in files:
        r = model.predict(source=str(p), imgsz=imgsz, verbose=False)[0]
        probs = r.probs
        # 类名到索引的映射由模型自带，不假设目录字母序
        idx = [i for i, n in model.names.items() if n == TRUE_NG]
        if not idx:
            raise SystemExit(f"模型的类别里没有 {TRUE_NG}，实际类别：{model.names}")
        y_score.append(float(probs.data[idx[0]]))
        y_true.append(1 if p.parent.name == TRUE_NG else 0)
        names.append(p.name)
    return np.array(y_true), np.array(y_score), names


def cost_curve(y_true: np.ndarray, y_score: np.ndarray, c_fn: float, c_fp: float):
    thresholds = np.unique(np.round(np.concatenate([[0.0, 0.5, 1.0], y_score]), 4))
    rows = []
    n_pos = max(int((y_true == 1).sum()), 1)
    n_neg = max(int((y_true == 0).sum()), 1)
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        fn = int(((y_true == 1) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        rows.append(dict(
            threshold=float(t),
            fn=fn, fp=fp,
            fnr=fn / n_pos, fpr=fp / n_neg,
            cost=(c_fn * fn + c_fp * fp) / n_pos,   # 归一化：每张真缺陷的平均期望代价
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ckpt = args.ckpt
    if ckpt is None:
        # 优先读 train.py 落盘的绝对路径，避免 ultralytics 的任务子目录导致路径不一致
        rec = Path("runs/last_ckpt.txt")
        ckpt = Path(rec.read_text(encoding="utf-8").strip()) if rec.exists() else \
            Path(cfg["model"]["project"]) / cfg["model"]["name"] / "weights/best.pt"
    test_dir = Path(cfg["data"]["root"]) / "test"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(ckpt))
    y_true, y_score, _ = collect(model, test_dir, cfg["data"]["imgsz"])
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    print(f"[eval] 测试集 n={len(y_true)}  真缺陷={n_pos}  伪缺陷={n_neg}  正例占比={n_pos/len(y_true):.3f}")

    auc = roc_auc_score(y_true, y_score)
    ap_ = average_precision_score(y_true, y_score)

    c_fn, c_fp = cfg["cost"]["c_fn"], cfg["cost"]["c_fp"]
    rows = cost_curve(y_true, y_score, c_fn, c_fp)
    best = min(rows, key=lambda r: r["cost"])
    at_half = next((r for r in rows if abs(r["threshold"] - 0.5) < 1e-9), None)

    metrics = {
        "n_test": len(y_true),
        "n_true_ng": n_pos,
        "n_pseudo_ng": n_neg,
        "pos_ratio": round(n_pos / len(y_true), 4),
        "auc": round(float(auc), 4),
        "average_precision": round(float(ap_), 4),
        "cost_model": {"c_fn": c_fn, "c_fp": c_fp},
        "best_threshold": best["threshold"],
        "at_best_threshold": {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in best.items()},
        "at_threshold_0.5": None if at_half is None else
                            {k: round(v, 4) if isinstance(v, float) else v for k, v in at_half.items()},
        # 朴素基线：全判伪（这是 accuracy 类指标失效的根源，必须显式给出）
        "baseline_all_pseudo": {"accuracy": round(n_neg / len(y_true), 4), "fnr": 1.0},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 图 1：PR 曲线 ----
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec = tpr * n_pos / np.maximum(tpr * n_pos + fpr * n_neg, 1e-9)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(tpr, prec, lw=1.8)
    ax[0].set_xlabel("Recall (true NG caught)")
    ax[0].set_ylabel("Precision")
    ax[0].set_title(f"PR curve  AP={ap_:.4f}")
    ax[0].grid(alpha=0.3)

    # ---- 图 2：代价 vs 阈值 ----
    th = [r["threshold"] for r in rows]
    ax[1].plot(th, [r["cost"] for r in rows], lw=1.8, label="expected cost")
    ax[1].axvline(best["threshold"], ls="--", c="g", label=f"tau*={best['threshold']:.2f}")
    ax[1].axvline(0.5, ls=":", c="gray", label="naive tau=0.5")
    ax[1].set_xlabel("threshold on P(true NG)")
    ax[1].set_ylabel(f"cost / defect  (C_fn={c_fn:g}, C_fp={c_fp:g})")
    ax[1].set_title("Cost-sensitive threshold")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "eval_curves.png", dpi=150)

    print(f"[eval] AUC={auc:.4f} AP={ap_:.4f}  tau*={best['threshold']:.2f} "
          f"FNR={best['fnr']:.4f} FPR={best['fpr']:.4f}")
    print(f"[eval] 全判伪基线 accuracy={n_neg/len(y_true):.4f}（说明 accuracy 不可单独使用）")
    print(f"[eval] 写出 {out/'metrics.json'} 与 {out/'eval_curves.png'}")


if __name__ == "__main__":
    main()
