"""Взвешенный стегоанализ, WS (Fridrich, Goljan, 2004; Ker, Böhme, 2008).

p = 2 * sum_i w_i (s_i - s̄_i)(s_i - ĉ_i),
где s̄_i - отсчёт с инвертированным LSB, ĉ_i - предсказание по 4 соседям,
w_i ~ 1 / (5 + σ_i^2) - веса по локальной дисперсии, sum w_i = 1.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def ws_analysis(ch: np.ndarray, max_value: int = 255) -> Dict:
    s = ch.astype(np.float64)
    if s.shape[0] < 3 or s.shape[1] < 3:
        return {"estimate": 0.0}
    up, down = s[:-2, 1:-1], s[2:, 1:-1]
    left, right = s[1:-1, :-2], s[1:-1, 2:]
    x = s[1:-1, 1:-1]
    nb = np.stack([up, down, left, right])
    pred = nb.mean(axis=0)
    var = nb.var(axis=0)
    # отсчёты на границах диапазона исключаются (для них flip несимметричен)
    valid = (x > 0) & (x < max_value)
    w = np.where(valid, 1.0 / (5.0 + var), 0.0)
    total = w.sum()
    if total <= 0:
        return {"estimate": 0.0}
    w /= total
    x_bar = x + 1.0 - 2.0 * np.mod(x, 2)
    p = 2.0 * np.sum(w * (x - x_bar) * (x - pred))
    return {"estimate": float(np.clip(p, 0.0, 1.0)), "raw": float(p)}
