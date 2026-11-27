"""伪缺陷合成：把「好产品」变成「被 AOI 误报的 NG」。

设计约束（这一段比代码重要）：
1. 扰动必须对应**现场真实诱因**，不是随便加噪声。产线上导致过杀的就那几类：
   光照不均、侧边反光、料带偏移造成的阴影、镜头脏污、传输压缩。
2. 必须保留一部分「未扰动的正常图」作为伪缺陷样本。否则模型会退化成
   "检测有没有扰动痕迹" 的二分类器，在真实现场立刻失效（shortcut learning）。
3. 扰动强度要随机化，且和真缺陷的形态有重叠区 —— 完全可分的数据集没有意义。
"""
from __future__ import annotations

import numpy as np
import cv2

# 每类扰动的采样概率。现场出现频率的量级估计：
# 光照/反光最常见，压缩是次生问题，镜头脏污偶发。
DEGRADE_PROBS = {
    "none": 0.22,        # 原样保留，什么都不做（防 shortcut）
    "illumination": 0.26,
    "glare": 0.18,
    "side_shadow": 0.16,
    "blur": 0.08,
    "jpeg": 0.06,
    "dust": 0.04,
}


def illumination_gradient(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """环形灯/条形光造成的亮度梯度。"""
    h, w = img.shape[:2]
    ang = rng.uniform(0, 2 * np.pi)
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    ramp = np.cos(ang) * xx + np.sin(ang) * yy
    gain = 1.0 + rng.uniform(0.25, 0.85) * ramp
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def glare_spot(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """镜面反光：一小块过曝区域，边缘柔和。"""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    cy, cx = rng.integers(0, h), rng.integers(0, w)
    r = rng.uniform(0.12, 0.32) * min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2)))
    out = out + mask * rng.uniform(60, 150)
    return np.clip(out, 0, 255).astype(np.uint8)


def side_shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """侧边遮挡/料带偏移造成的暗带。"""
    h, w = img.shape[:2]
    side = rng.integers(0, 4)
    frac = rng.uniform(0.06, 0.22)
    out = img.astype(np.float32)
    if side == 0:
        out[: int(h * frac)] *= rng.uniform(0.35, 0.7)
    elif side == 1:
        out[-int(h * frac):] *= rng.uniform(0.35, 0.7)
    elif side == 2:
        out[:, : int(w * frac)] *= rng.uniform(0.35, 0.7)
    else:
        out[:, -int(w * frac):] *= rng.uniform(0.35, 0.7)
    return np.clip(out, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """对焦不足 / 运动模糊。"""
    k = int(rng.choice([3, 5, 7]))
    if rng.random() < 0.5:
        return cv2.GaussianBlur(img, (k, k), 0)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def jpeg(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """传输/存储压缩。"""
    q = int(rng.integers(35, 75))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if img.ndim == 2 else cv2.IMREAD_COLOR)


def dust(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """镜头脏污：若干模糊暗斑。"""
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(int(rng.integers(2, 6))):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        r = int(rng.uniform(0.02, 0.07) * min(h, w))
        overlay = out.copy()
        cv2.circle(overlay, (int(cx), int(cy)), r, int(0), -1)
        out = cv2.addWeighted(overlay, rng.uniform(0.25, 0.55), out, 1 - 0.35, 0)
    return out


_OPS = {
    "illumination": illumination_gradient,
    "glare": glare_spot,
    "side_shadow": side_shadow,
    "blur": blur,
    "jpeg": jpeg,
    "dust": dust,
}


def degrade(img: np.ndarray, rng: np.random.Generator, kind: str | None = None) -> tuple[np.ndarray, str]:
    """按概率表施加一种扰动（或指定）。返回 (图, 扰动名)。"""
    if kind is None:
        keys = list(DEGRADE_PROBS)
        probs = np.array([DEGRADE_PROBS[k] for k in keys], dtype=np.float64)
        kind = str(rng.choice(keys, p=probs / probs.sum()))
    if kind == "none":
        return img, "none"
    return _OPS[kind](img, rng), kind


if __name__ == "__main__":
    # 自检：生成一张示例，确认各算子不会崩
    rng = np.random.default_rng(0)
    demo = (rng.normal(128, 40, (300, 300))).clip(0, 255).astype(np.uint8)
    for k in _OPS:
        out, name = degrade(demo, rng, k)
        print(f"{name:14s} -> shape={out.shape} dtype={out.dtype} mean={out.mean():.1f}")
