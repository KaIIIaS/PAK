"""Оценка стеганографической стойкости PNG-изображения как контейнера.

Методика: в изображение (центральный фрагмент) последовательно встраиваются
случайные сообщения с возрастающей долей заполнения r (бит/отсчёт) методами
LSB-замены и LSB-согласования (±1). Для каждого стегообраза ансамбль «слепых»
детекторов вычисляет вероятность обнаружения P_d(r). По результатам определяются:
  * безопасная доля заполнения r* - максимальная r, при которой P_d < порога
    для всех r' <= r;
  * безопасная ёмкость C* = r* · N / 8 байт (N - число отсчётов полного изображения);
  * индекс стеганографической стойкости ИСС = 100 · (1 - AUC), где AUC - нормированная
    площадь под кривой P_d(log r).
Высокий ИСС означает, что контейнер позволяет скрыть значительный объём данных
незаметно для средств контроля (высокий риск для защищаемой системы).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .analysis.ensemble import quick_scores
from .config import Config
from .embedding import embed_random_rate
from .image_io import ImageData
from .metrics import channel_metrics

METHOD_LABELS = {"lsb_replacement": "LSB-замена", "lsb_matching": "LSB-согласование (±1)"}


def stability_class(index: float) -> str:
    if index >= 70:
        return "высокая"
    if index >= 40:
        return "средняя"
    return "низкая"


def assess_robustness(img: ImageData, cfg: Optional[Config] = None, progress=None) -> Dict:
    cfg = cfg or Config()
    rs_cfg, thr = cfg.robustness, cfg.thresholds
    if img.is_palette:
        return {"supported": False, "reason": "Палитровые изображения не оцениваются (LSB-методы неприменимы)."}
    work = img.crop_side(rs_cfg.max_side)
    cover = {k: v for k, v in work.channels.items() if k != "A"}
    rng = np.random.default_rng(cfg.seed)
    baseline = quick_scores(cover, work.max_value, thr)
    rates = sorted(rs_cfg.rates)
    n_full = sum(v.size for k, v in img.channels.items() if k != "A")
    total_steps = len(rs_cfg.methods) * len(rates) * rs_cfg.trials
    step = 0
    methods = {}
    for method in rs_cfg.methods:
        points = []
        for rate in rates:
            pds, pays, ps, ss = [], [], [], []
            for _ in range(rs_cfg.trials):
                stego = embed_random_rate(cover, rate, method, rng, work.max_value)
                q = quick_scores(stego, work.max_value, thr)
                m = channel_metrics(cover, stego, work.max_value)
                pds.append(q["detection_probability"])
                pays.append(q["payload_estimate"])
                ps.append(m["psnr"])
                ss.append(m["ssim"])
                step += 1
                if progress:
                    progress(step / total_steps)
            pd = float(np.mean(pds))
            points.append({"rate": rate, "p_detect": pd, "payload_estimate": float(np.mean(pays)),
                           "detected": pd >= rs_cfg.detection_threshold,
                           "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))})
        safe = 0.0
        for pt in points:
            if pt["detected"]:
                break
            safe = pt["rate"]
        if len(rates) > 1:
            x = np.log(np.array(rates))
            y = np.array([p["p_detect"] for p in points])
            auc = float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)) / (x[-1] - x[0]))
        else:
            auc = points[0]["p_detect"]
        index = float(np.clip(100 * (1 - auc), 0, 100))
        first_detect = next((p["rate"] for p in points if p["detected"]), None)
        methods[method] = {
            "label": METHOD_LABELS.get(method, method),
            "points": points,
            "safe_rate": safe,
            "first_detected_rate": first_detect,
            "secure_capacity_bytes": int(safe * n_full / 8),
            "stability_index": index,
            "stability_class": stability_class(index),
        }
    worst = max(methods.values(), key=lambda m: m["stability_index"])
    overall = worst["stability_index"]
    notes = []
    if baseline["detection_probability"] >= rs_cfg.detection_threshold:
        notes.append("Исходное изображение уже вызывает срабатывание детекторов: оценка стойкости "
                     "может быть занижена (возможно, изображение уже содержит вложение).")
    return {
        "supported": True,
        "fragment": f"{work.width}x{work.height}",
        "baseline": baseline,
        "methods": methods,
        "stability_index": overall,
        "stability_class": stability_class(overall),
        "most_dangerous_method": worst["label"],
        "max_secure_capacity_bytes": max(m["secure_capacity_bytes"] for m in methods.values()),
        "total_capacity_bytes": int(n_full / 8),
        "notes": notes,
    }
