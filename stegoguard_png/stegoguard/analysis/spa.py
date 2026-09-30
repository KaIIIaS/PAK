"""Анализ пар отсчётов, SPA (Dumitrescu, Wu, Wang, 2003).

Для пар соседних отсчётов (u, v) формируются множества:
  X: (v чётное и u < v) или (v нечётное и u > v);
  Y: (v чётное и u > v) или (v нечётное и u < v);
  W+Z: пары с floor(u/2) = floor(v/2) (в том числе u = v); P - все пары.
Доля вложения p - меньший корень уравнения
    0.5 (|W|+|Z|) p^2 + (2|X| - |P|) p + |Y| - |X| = 0.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _estimate(u: np.ndarray, v: np.ndarray) -> float:
    v_even = (v & 1) == 0
    x = np.count_nonzero((v_even & (u < v)) | (~v_even & (u > v)))
    y = np.count_nonzero((v_even & (u > v)) | (~v_even & (u < v)))
    wz = np.count_nonzero((u >> 1) == (v >> 1))
    p_all = u.size
    a = 0.5 * wz
    b = 2.0 * x - p_all
    c = float(y - x)
    if a < 1e-12:
        return float(np.clip(-c / b, 0, 1)) if abs(b) > 1e-12 else 0.0
    disc = b * b - 4 * a * c
    if disc < 0:
        return float(np.clip(-b / (2 * a), 0, 1))
    sq = disc ** 0.5
    roots = sorted([(-b + sq) / (2 * a), (-b - sq) / (2 * a)])
    return float(np.clip(roots[0], 0.0, 1.0))


def spa_analysis(ch: np.ndarray) -> Dict:
    ch = ch.astype(np.int64)
    ests = []
    if ch.shape[1] > 1:
        ests.append(_estimate(ch[:, :-1].ravel(), ch[:, 1:].ravel()))
    if ch.shape[0] > 1:
        ests.append(_estimate(ch[:-1, :].ravel(), ch[1:, :].ravel()))
    return {"estimate": float(np.mean(ests)) if ests else 0.0,
            "estimates_by_axis": [float(e) for e in ests]}
