"""旁路服务：AOI 调我，我不调 AOI。

接口契约（这条比代码重要）：调用方永远拿到一个 200，永远不会因为我的服务挂掉
而卡住产线。出错时返回 verdict="pass_through"，让产线退回原来的人工复核流程。
这是"零侵入"的最后一公里：不是不报错，是报错了也不能拖累客户。
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO

CFG = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
THRESHOLD = float(CFG["serve"]["threshold"])
_rec = Path("runs/last_ckpt.txt")
CKPT = _rec.read_text(encoding="utf-8").strip() if _rec.exists() else CFG["serve"]["ckpt"]

app = FastAPI(title="AOI Pseudo-Defect Recheck", version="0.1.0")
_model: YOLO | None = None
_stats = {"total": 0, "true_ng": 0, "pseudo_ng": 0, "pass_through": 0}


def model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(CKPT)
    return _model


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "ckpt": CKPT, "threshold": THRESHOLD}


@app.get("/metrics")
def metrics() -> dict:
    return _stats


@app.post("/recheck")
async def recheck(file: UploadFile = File(...)) -> dict:
    t0 = time.perf_counter()
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("L")
        arr = np.array(img)[:, :, None]                      # HWC，ultralytics 接受
        r = model().predict(source=arr, imgsz=CFG["data"]["imgsz"], verbose=False)[0]
        names = model().names
        idx = [i for i, n in names.items() if n == "true_ng"][0]
        p_true = float(r.probs.data[idx])
        verdict = "true_ng" if p_true >= THRESHOLD else "pseudo_ng"
        _stats["total"] += 1
        _stats[verdict] += 1
        return {
            "verdict": verdict,
            "p_true_ng": round(p_true, 4),
            "threshold": THRESHOLD,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
    except Exception as e:  # noqa: BLE001 —— 故意兜住，产线不能被拖死
        _stats["total"] += 1
        _stats["pass_through"] += 1
        return {"verdict": "pass_through", "error": str(e),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
