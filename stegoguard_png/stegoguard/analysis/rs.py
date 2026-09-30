"""RS-стегоанализ (Fridrich, Goljan, Du, 2001).

Пиксели группируются по 4; функция дискриминации f(G) = sum |x_{i+1} - x_i|.
Для маски M и инвертированной -M подсчитываются доли регулярных (R) и сингулярных (S)
групп на исходном изображении и на изображении с инвертированными LSB. Доля
вложения p находится из квадратного уравнения
    2(d1 + d0) z^2 + (d-0 - d-1 - d1 - 3 d0) z + d0 - d-0 = 0,  p = z / (z - 1/2).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

DEFAULT_MASK = (0, 1, 1, 0)


def _groups(ch: np.ndarray, axis: int) -> np.ndarray:
    arr = ch if axis == 1 else ch.T
    w4 = (arr.shape[1] // 4) * 4
    if w4 == 0:
        return np.empty((0, 4), dtype=np.int64)
    return np.ascontiguousarray(arr[:, :w4]).reshape(-1, 4).astype(np.int64)


def _disc(g: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(g, axis=1)).sum(axis=1)


def _apply_mask(g: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = g.copy()
    pos = np.flatnonzero(mask == 1)
    neg = np.flatnonzero(mask == -1)
    if pos.size:
        out[:, pos] ^= 1                              # F1: 0<->1, 2<->3, ...
    if neg.size:
        out[:, neg] = ((out[:, neg] + 1) ^ 1) - 1     # F-1: -1<->0, 1<->2, ...
    return out


def _rs_stats(g: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, float]:
    f0 = _disc(g)
    fp = _disc(_apply_mask(g, mask))
    fn = _disc(_apply_mask(g, -mask))
    n = float(len(f0))
    return (np.count_nonzero(fp > f0) / n, np.count_nonzero(fp < f0) / n,
            np.count_nonzero(fn > f0) / n, np.count_nonzero(fn < f0) / n)


def _solve(s0, s1) -> float:
    rm, sm, rnm, snm = s0
    rm1, sm1, rnm1, snm1 = s1
    d0, d1 = rm - sm, rm1 - sm1
    dn0, dn1 = rnm - snm, rnm1 - snm1
    a = 2.0 * (d1 + d0)
    b = dn0 - dn1 - d1 - 3.0 * d0
    c = d0 - dn0
    if abs(a) < 1e-12:
        z = -c / b if abs(b) > 1e-12 else 0.0
    else:
        disc = max(b * b - 4 * a * c, 0.0)
        sq = disc ** 0.5
        z1, z2 = (-b + sq) / (2 * a), (-b - sq) / (2 * a)
        z = z1 if abs(z1) < abs(z2) else z2
    if abs(z - 0.5) < 1e-12:
        return 1.0
    return float(np.clip(z / (z - 0.5), 0.0, 1.0))


def rs_analysis(ch: np.ndarray, mask=DEFAULT_MASK) -> Dict:
    m = np.asarray(mask, dtype=np.int64)
    estimates = []
    details: Dict = {}
    for axis in (1, 0):
        g = _groups(ch, axis)
        if g.shape[0] < 64:
            continue
        s0 = _rs_stats(g, m)
        s1 = _rs_stats(g ^ 1, m)
        estimates.append(_solve(s0, s1))
        if not details:
            details = {"Rm": s0[0], "Sm": s0[1], "R-m": s0[2], "S-m": s0[3]}
    est = float(np.mean(estimates)) if estimates else 0.0
    return {"estimate": est, "estimates_by_axis": [float(e) for e in estimates], **{k: float(v) for k, v in details.items()}}
