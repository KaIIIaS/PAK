"""Атака хи-квадрат Вестфельда–Пфитцмана (анализ пар значений, PoV).

При LSB-замене частоты значений 2k и 2k+1 выравниваются. Статистика
    X^2 = sum_k (h[2k] - (h[2k]+h[2k+1])/2)^2 / ((h[2k]+h[2k+1])/2)
при гипотезе о наличии вложения имеет распределение хи-квадрат; p = 1 - F(X^2).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.stats import chi2


def chi_square_pov(values: np.ndarray, max_value: int = 255, min_expected: float = 5.0) -> Dict:
    nb = max_value + 1 + ((max_value + 1) % 2)
    h = np.bincount(np.asarray(values).ravel().astype(np.int64), minlength=nb)[:nb].astype(np.float64)
    even, odd = h[0::2], h[1::2]
    expected = (even + odd) / 2.0
    mask = expected > min_expected
    k = int(mask.sum())
    if k < 2:
        return {"chi2": 0.0, "df": 0, "p_value": 0.0}
    stat = float(np.sum((even[mask] - expected[mask]) ** 2 / expected[mask]))
    df = k - 1
    return {"chi2": stat, "df": df, "p_value": float(chi2.sf(stat, df))}


def progressive_chi_square(channel: np.ndarray, max_value: int = 255, steps: int = 20) -> List[Dict]:
    """Кривая p-значения по нарастающей доле изображения (порядок развёртки строк)."""
    flat = np.asarray(channel).ravel()
    n = flat.size
    curve = []
    for i in range(1, steps + 1):
        part = flat[: max(2, int(n * i / steps))]
        curve.append({"fraction": i / steps, "p_value": chi_square_pov(part, max_value)["p_value"]})
    return curve


def sequential_length(curve: List[Dict], threshold: float = 0.95) -> float:
    """Оценка длины последовательного вложения: наибольшая начальная доля с p > порога."""
    length = 0.0
    for point in curve:
        if point["p_value"] > threshold:
            length = point["fraction"]
        else:
            break
    return length
