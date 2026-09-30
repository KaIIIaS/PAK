"""Вспомогательные признаки: статистика LSB-плоскости, шум, сложность текстуры, HCF-COM."""
from __future__ import annotations

from typing import Dict

import numpy as np


def lsb_plane_stats(ch: np.ndarray) -> Dict:
    lsb = (ch & 1).astype(np.int64)
    h, w = (lsb.shape[0] // 2) * 2, (lsb.shape[1] // 2) * 2
    ent = 0.0
    if h and w:
        blocks = (lsb[0:h:2, 0:w:2] << 3) | (lsb[1:h:2, 0:w:2] << 2) | (lsb[0:h:2, 1:w:2] << 1) | lsb[1:h:2, 1:w:2]
        cnt = np.bincount(blocks.ravel(), minlength=16).astype(np.float64)
        p = cnt[cnt > 0] / cnt.sum()
        ent = float(-(p * np.log2(p)).sum() / 4.0)
    agree_h = float(np.mean(lsb[:, 1:] == lsb[:, :-1])) if lsb.shape[1] > 1 else 0.5
    agree_v = float(np.mean(lsb[1:, :] == lsb[:-1, :])) if lsb.shape[0] > 1 else 0.5
    return {"ones_ratio": float(lsb.mean()), "block_entropy": ent,
            "neighbor_agreement": (agree_h + agree_v) / 2}


def noise_sigma(ch: np.ndarray) -> float:
    """Быстрая оценка СКО шума (Immerkaer, 1996)."""
    s = ch.astype(np.float64)
    if s.shape[0] < 3 or s.shape[1] < 3:
        return 0.0
    conv = (s[:-2, :-2] - 2 * s[:-2, 1:-1] + s[:-2, 2:]
            - 2 * s[1:-1, :-2] + 4 * s[1:-1, 1:-1] - 2 * s[1:-1, 2:]
            + s[2:, :-2] - 2 * s[2:, 1:-1] + s[2:, 2:])
    h, w = s.shape
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))


def hcf_com_ratio(ch: np.ndarray, max_value: int = 255) -> float:
    """Калиброванный центр масс характеристической функции гистограммы (Harmsen; Ker).
    Отношение COM(изображение)/COM(прореженное 2x2). Снижение отношения характерно
    для LSB-matching (±1), которое действует как низкочастотный фильтр гистограммы."""
    if max_value != 255 or ch.shape[0] < 4 or ch.shape[1] < 4:
        return 1.0

    def com(values: np.ndarray) -> float:
        hist = np.bincount(values.ravel().astype(np.int64), minlength=256)[:256].astype(np.float64)
        spec = np.abs(np.fft.fft(hist))[1:129]
        k = np.arange(1, 129)
        return float((k * spec).sum() / max(spec.sum(), 1e-12))

    h, w = (ch.shape[0] // 2) * 2, (ch.shape[1] // 2) * 2
    c = ch[:h, :w].astype(np.float64)
    down = np.rint((c[0::2, 0::2] + c[1::2, 0::2] + c[0::2, 1::2] + c[1::2, 1::2]) / 4.0)
    base = com(down)
    return float(com(ch) / base) if base > 0 else 1.0


def image_complexity(channels: Dict[str, np.ndarray], max_value: int = 255) -> Dict:
    """Характеристики изображения, влияющие на стеганографическую стойкость контейнера."""
    sig, grad, sat, uniq = [], [], [], []
    for ch in channels.values():
        c = ch.astype(np.float64)
        sig.append(noise_sigma(ch))
        gx = np.abs(np.diff(c, axis=1)).mean() if c.shape[1] > 1 else 0.0
        gy = np.abs(np.diff(c, axis=0)).mean() if c.shape[0] > 1 else 0.0
        grad.append((gx + gy) / 2)
        sat.append(float(np.mean((ch == 0) | (ch == max_value))))
        uniq.append(int(np.unique(ch).size))
    scale = max_value / 255.0
    noise = float(np.mean(sig)) / scale
    texture = float(np.mean(grad)) / scale
    # Нормированный индекс сложности 0..1: шумные текстурированные изображения
    # лучше маскируют изменения LSB.
    index = float(np.clip(0.5 * min(noise / 4.0, 1.0) + 0.5 * min(texture / 12.0, 1.0), 0, 1))
    return {"noise_sigma": noise, "mean_gradient": texture,
            "saturation_ratio": float(np.mean(sat)), "unique_levels_min": int(min(uniq)),
            "complexity_index": index,
            "complexity_label": "высокая" if index > 0.6 else ("средняя" if index > 0.3 else "низкая")}


def _dominant_levels(ch: np.ndarray, share: float = 0.002) -> int:
    counts = np.bincount(ch.ravel().astype(np.int64))
    return int(np.count_nonzero(counts > share * ch.size))


def flat_region_test(ch: np.ndarray, max_value: int = 255, min_flat_fraction: float = 0.02) -> Dict:
    """Тест однородных областей (FRN) для синтетического контента (экраны HMI/SCADA, мнемосхемы,
    схемы). В чистом синтетическом изображении окна 3x3 с размахом значений <= 2 практически всегда
    постоянны. Любое LSB-вложение с долей r изменяет каждый отсчёт с вероятностью r/2, поэтому
    доля непостоянных окон f = 1 - (1 - r/2)^9, откуда r = 2 (1 - (1 - f)^(1/9)).
    Вариант А (строгий): окна с 8 одинаковыми соседями, отклонение центра ровно на ±1 (оценка 2·доля).
    Вариант Б (почти однородные окна) применяется только к изображениям с малым числом
    доминирующих уровней (синтетический контент), т.к. на малошумных фото даёт ложные срабатывания."""
    res = {"applicable": False, "flat_fraction": 0.0, "estimate": 0.0}
    if ch.shape[0] < 3 or ch.shape[1] < 3:
        return res
    c = ch[:-2, :-2]
    x = ch[1:-1, 1:-1]
    win = [ch[i:ch.shape[0] - 2 + i, j:ch.shape[1] - 2 + j] for i in range(3) for j in range(3)]
    nbs = [w for k, w in enumerate(win) if k not in (0, 4)]
    flat = (c > 0) & (c < max_value)
    for nb in nbs:
        flat &= nb == c
    n_flat = int(np.count_nonzero(flat))
    frac = n_flat / flat.size
    res["flat_fraction"] = float(frac)
    estimates = []
    if frac >= min_flat_fraction and n_flat >= 500:
        dev = int(np.count_nonzero(flat & (np.abs(x - c) == 1)))
        res["deviation_ratio"] = dev / n_flat
        estimates.append(float(np.clip(2.0 * dev / n_flat, 0.0, 1.0)))
    dom = _dominant_levels(ch)
    res["dominant_levels"] = dom
    if dom <= 48:
        stack = np.stack(win)
        lo, hi = stack.min(axis=0), stack.max(axis=0)
        near = (hi - lo <= 2) & (lo > 0) & (hi < max_value)
        n_near = int(np.count_nonzero(near))
        if n_near >= 500 and n_near / near.size >= 0.05:
            f = np.count_nonzero(near & (hi != lo)) / n_near
            res["nonconstant_ratio"] = float(f)
            estimates.append(float(np.clip(2.0 * (1.0 - (1.0 - f) ** (1.0 / 9.0)), 0.0, 1.0)))
    if estimates:
        res["applicable"] = True
        res["estimate"] = max(estimates)
    return res
