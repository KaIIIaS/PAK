"""Метрики искажения контейнера: MSE, PSNR, SSIM."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_value: int = 255) -> float:
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(10 * np.log10(max_value ** 2 / m))


def ssim(a: np.ndarray, b: np.ndarray, max_value: int = 255, sigma: float = 1.5) -> float:
    x, y = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * max_value) ** 2, (0.03 * max_value) ** 2
    mx, my = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
    sxx = gaussian_filter(x * x, sigma) - mx * mx
    syy = gaussian_filter(y * y, sigma) - my * my
    sxy = gaussian_filter(x * y, sigma) - mx * my
    num = (2 * mx * my + c1) * (2 * sxy + c2)
    den = (mx * mx + my * my + c1) * (sxx + syy + c2)
    return float(np.mean(num / den))


def channel_metrics(cover: dict, stego: dict, max_value: int = 255) -> dict:
    ps = [psnr(cover[k], stego[k], max_value) for k in cover]
    ss = [ssim(cover[k], stego[k], max_value) for k in cover]
    finite = [p for p in ps if np.isfinite(p)]
    return {"psnr": float(np.mean(finite)) if finite else float("inf"),
            "ssim": float(np.mean(ss)), "mse": float(np.mean([mse(cover[k], stego[k]) for k in cover]))}
