"""把 ELPV 公开数据集重建成「过杀复判」二分类数据集。

映射关系（对应真实项目）：
    ELPV defect      -> true_ng    (真缺陷，AOI 判 NG 判对了)
    ELPV functional  -> pseudo_ng  (好产品被 AOI 误报，需要复判拦下来)

关键点：**按拍摄批次（≈组件）划分 train/val/test**。
同一块组件切出来的电池片外观高度相似，随机切图会让测试集泄漏，指标虚高。
这是工业视觉数据里最常见、也最容易被面试官抓的一个坑。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from synth_pseudo import degrade


def parse_labels(elpv_root: Path, image_files: list[Path]) -> dict[str, int]:
    """读出 文件名 -> 标签(0=functional, 1=defect)。

    ELPV 的 labels.csv 是空白分隔的三列：image_path, defect_probability, cell_type。
    没有二值标签，阈值取 0.5（数据集标注惯例）。
    也兼容早期的逗号分隔 + prediction 列版本。
    """
    csv_path = elpv_root / "labels.csv"
    text = [ln for ln in csv_path.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
    header_raw = text[0].strip()
    delim = "," if "," in header_raw else None      # None = 任意空白
    cells_ = (lambda s: [c.strip() for c in (s.split(",") if delim else s.split())])
    first = cells_(header_raw)

    # 该版本 labels.csv 没有表头（首行即数据），用"首列像不像文件路径"判断
    has_header = not any(tok in first[0].lower() for tok in (".png", ".jpg", "/", "\\"))
    header = [h.lower() for h in first] if has_header else ["path", "probability", "type"]
    body = text[1:] if has_header else text
    cells = cells_

    # 分列：文件名列 / 概率列 / 二值列
    name_i = next((i for i, h in enumerate(header) if h in {"path", "file", "filename", "name"}), 0)
    prob_i = next((i for i, h in enumerate(header) if "prob" in h), None)
    pred_i = next((i for i, h in enumerate(header) if h == "prediction"), None)

    out: dict[str, int] = {}
    for ln in body:
        c = cells(ln)
        if len(c) <= max(name_i, prob_i or 0):
            continue
        name = Path(c[name_i]).name
        if pred_i is not None:
            out[name] = int(float(c[pred_i]))
        elif prob_i is not None:
            out[name] = int(float(c[prob_i]) >= 0.5)
        else:
            raise SystemExit(f"labels.csv 既没有 prediction 也没有 probability 列：{header}")

    missing = [p.name for p in image_files if p.name not in out]
    if missing:
        raise SystemExit(f"{len(missing)} 张图没有标签，例如 {missing[:3]}")
    return out


def batch_key(path: Path, index: int, block: int = 50) -> str:
    """划分用的"批次"键。

    ELPV 文件名是 cell0001.png 这种顺序编号，不带组件号，无法直接按组件分组。
    但它是按采集顺序枚举的，相邻编号来自同一块组件 —— 所以用**连续编号块**近似组件，
    保证同一块的图整组进同一个 split，避免测试集里混进训练集"同款"造成指标虚高。
    若文件名里带拍摄时间戳（原始产线数据常见），优先用时间戳分组。
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", path.stem)
    return m.group(1) if m else f"block{index // block:04d}"


def split_groups(groups: list[str], seed: int, ratios=(0.7, 0.15, 0.15)) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    order = np.array(sorted(groups), dtype=object)
    rng.shuffle(order)
    n = len(order)
    n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
    assign = {}
    for g in order[:n_tr]:
        assign[g] = "train"
    for g in order[n_tr:n_tr + n_va]:
        assign[g] = "val"
    for g in order[n_tr + n_va:]:
        assign[g] = "test"
    return assign


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path,
                    default=Path("data/raw/elpv-dataset-master/src/elpv_dataset/data"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pseudo-ratio", type=float, default=2.0,
                    help="伪缺陷样本数 / 真缺陷样本数 的目标比例。现场过杀多于此，这里取保守值。")
    args = ap.parse_args()

    img_dir = args.raw / "images"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        raise SystemExit(f"没找到图片：{img_dir}")
    labels = parse_labels(args.raw, images)
    print(f"[data] 图片 {len(images)} 张，标签 {len(labels)} 条，"
          f"defect={sum(labels.values())} functional={len(labels) - sum(labels.values())}")

    groups = sorted({batch_key(p, i) for i, p in enumerate(images)})
    assign = split_groups(groups, args.seed)
    print(f"[data] 批次分组 {len(groups)} 个 -> "
          f"{ {s: sum(1 for v in assign.values() if v == s) for s in ('train', 'val', 'test')} }")

    for split in ("train", "val", "test"):
        for cls in ("true_ng", "pseudo_ng"):
            (args.out / split / cls).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    rng = np.random.default_rng(args.seed)
    counters: dict[tuple[str, str], int] = defaultdict(int)

    split_of = {p.name: assign[batch_key(p, i)] for i, p in enumerate(images)}
    idx_of = {p.name: i for i, p in enumerate(images)}
    defect_by_split = {s: 0 for s in ("train", "val", "test")}
    for p in images:
        if labels.get(p.name) == 1:
            defect_by_split[split_of[p.name]] += 1

    # ---- 真缺陷：原样拷贝 ----
    for i, p in enumerate(images):
        if labels.get(p.name) != 1:
            continue
        split = split_of[p.name]
        shutil.copy2(p, args.out / split / "true_ng" / p.name)
        counters[(split, "true_ng")] += 1
        rows.append(dict(split=split, cls="true_ng", batch=batch_key(p, i), src=p.name,
                         out=p.name, degrade="none"))

    # ---- 伪缺陷：按目标比例从 functional 中采样，逐个施加现场型退化 ----
    # 目标数 = ratio × 该 split 的真缺陷数。扰动类型由概率表决定，其中 "none"
    # 占约 22% —— 这部分"没被扰动的好产品"是关键：防止模型把任务退化成
    # "有没有扰动痕迹"的二分类（shortcut learning）。
    for split in ("train", "val", "test"):
        funcs = [p for p in images if labels.get(p.name) == 0 and split_of[p.name] == split]
        if not funcs:
            continue
        rng.shuffle(funcs)
        target = int(round(defect_by_split[split] * args.pseudo_ratio))
        for j in range(target):
            src = funcs[j % len(funcs)]
            rnd = j // len(funcs)
            img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            out_img, used = degrade(img, rng, None)
            name = f"{src.stem}__r{rnd}_{used}.png"
            cv2.imwrite(str(args.out / split / "pseudo_ng" / name), out_img)
            counters[(split, "pseudo_ng")] += 1
            rows.append(dict(split=split, cls="pseudo_ng", batch=batch_key(src, idx_of[src.name]),
                             src=src.name, out=name, degrade=used))

    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "cls", "batch", "src", "out", "degrade"])
        w.writeheader()
        w.writerows(rows)

    stats = {}
    for split in ("train", "val", "test"):
        t = counters.get((split, "true_ng"), 0)
        f_ = counters.get((split, "pseudo_ng"), 0)
        stats[split] = dict(true_ng=t, pseudo_ng=f_,
                            pos_ratio=round(t / max(t + f_, 1), 4))
    with open(args.out / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("[data] 完成：", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
