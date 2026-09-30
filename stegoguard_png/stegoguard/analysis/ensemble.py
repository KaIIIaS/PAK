"""Ансамбль детекторов и агрегирование оценок по каналам."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .chi_square import chi_square_pov, progressive_chi_square, sequential_length
from .features import flat_region_test, hcf_com_ratio, lsb_plane_stats, noise_sigma
from .rs import rs_analysis
from .spa import spa_analysis
from .ws import ws_analysis

MIN_LEVELS = 8


def detection_probability(payload: float, chi_p: float, thr) -> float:
    """Отображение оценок в вероятность наличия вложения (логистическая модель)."""
    mid = (thr.payload_suspicious + thr.payload_stego) / 2.0
    scale = max((thr.payload_stego - thr.payload_suspicious) / 4.0, 1e-4)
    p_payload = 1.0 / (1.0 + math.exp(-(payload - mid) / scale))
    p_chi = chi_p if chi_p >= thr.chi_square_p else 0.0
    return float(max(p_payload, p_chi))


def analyze_channel(ch: np.ndarray, max_value: int, full: bool = True, chi_threshold: float = 0.95) -> Dict:
    levels = int(np.unique(ch).size) if full else MIN_LEVELS
    if full and levels < MIN_LEVELS:
        return {"reliable": False, "reason": f"мало уровней яркости ({levels})",
                "payload_estimate": 0.0, "chi_square": {"p_value": 0.0}}
    rs = rs_analysis(ch)
    spa = spa_analysis(ch)
    ws = ws_analysis(ch, max_value)
    chi = chi_square_pov(ch, max_value)
    frn = flat_region_test(ch, max_value)
    ests = [rs["estimate"], spa["estimate"], ws["estimate"]]
    # WS наиболее несмещён на несжатых изображениях, RS и SPA повышают устойчивость
    payload = 0.5 * ws["estimate"] + 0.25 * rs["estimate"] + 0.25 * spa["estimate"]
    if frn["applicable"]:
        payload = max(payload, frn["estimate"])
    res = {
        "reliable": True,
        "rs": rs, "spa": spa, "ws": ws, "frn": frn, "chi_square": chi,
        "payload_estimate": float(payload),
        "estimator_spread": float(np.max(ests) - np.min(ests)),
    }
    if full:
        curve = progressive_chi_square(ch, max_value)
        res["progressive_chi"] = curve
        res["sequential_length"] = sequential_length(curve, chi_threshold)
        res["lsb"] = lsb_plane_stats(ch)
        res["hcf_com_ratio"] = hcf_com_ratio(ch, max_value)
        res["noise_sigma"] = noise_sigma(ch)
        res["levels"] = levels
    return res


def run_statistical(channels: Dict[str, np.ndarray], max_value: int, thr, full: bool = True,
                    palette: bool = False) -> Dict:
    per = {name: analyze_channel(ch, max_value, full, thr.chi_square_p) for name, ch in channels.items()}
    reliable = {k: v for k, v in per.items() if v.get("reliable")}
    payloads = [v["payload_estimate"] for v in reliable.values()]
    chi_ps = [v["chi_square"]["p_value"] for v in reliable.values()]
    payload = float(np.mean(payloads)) if payloads else 0.0
    payload_max = float(np.max(payloads)) if payloads else 0.0
    chi_max = float(np.max(chi_ps)) if chi_ps else 0.0
    # Вложение может быть сосредоточено в одном канале: учитываем максимум с весом
    effective = max(payload, 0.75 * payload_max)
    p_det = detection_probability(effective, chi_max, thr)
    seq = max((v.get("sequential_length", 0.0) for v in reliable.values()), default=0.0)
    notes = []
    if palette:
        notes.append("Палитровое изображение: анализ ведётся по индексам палитры, достоверность ниже.")
    if not reliable:
        notes.append("Недостаточно информативных каналов для статистического анализа.")
    return {
        "channels": per,
        "payload_estimate": payload,
        "payload_max_channel": payload_max,
        "effective_payload": effective,
        "chi_square_max_p": chi_max,
        "sequential_length": seq,
        "detection_probability": p_det,
        "reliable": bool(reliable) and not palette,
        "notes": notes,
    }


def quick_scores(channels: Dict[str, np.ndarray], max_value: int, thr) -> Dict:
    r = run_statistical(channels, max_value, thr, full=False)
    return {"payload_estimate": r["payload_estimate"], "effective_payload": r["effective_payload"],
            "chi_square_max_p": r["chi_square_max_p"], "detection_probability": r["detection_probability"]}
