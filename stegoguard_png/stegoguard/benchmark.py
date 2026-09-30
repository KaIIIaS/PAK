"""Оценка точности комплекса на размеченном наборе (manifest.json)."""
from __future__ import annotations

import json
import os
from typing import Dict

from .analyzer import Analyzer


def run_benchmark(dataset_dir: str, analyzer: Analyzer, suspicious_is_positive: bool = True) -> Dict:
    with open(os.path.join(dataset_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    tp = fp = tn = fn = 0
    rows = []
    for e in manifest:
        r = analyzer.analyze_file(os.path.join(dataset_dir, e["file"]))
        positive = r["verdict"] == "STEGO_DETECTED" or (suspicious_is_positive and r["verdict"] == "SUSPICIOUS")
        actual = e["label"] == "stego"
        tp += positive and actual
        fp += positive and not actual
        tn += (not positive) and not actual
        fn += (not positive) and actual
        rows.append({**e, "verdict": r["verdict"], "risk": r.get("risk_score"), "correct": positive == actual})
    total = max(len(manifest), 1)
    by_detail: Dict[str, Dict[str, int]] = {}
    for row in rows:
        key = row["detail"].split(",")[0] if row["label"] == "stego" else "чистые контейнеры"
        d = by_detail.setdefault(key, {"total": 0, "correct": 0})
        d["total"] += 1
        d["correct"] += int(row["correct"])
    return {
        "total": len(manifest), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / total,
        "tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "by_type": by_detail, "rows": rows,
    }
