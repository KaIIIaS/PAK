"""Интегральная оценка риска и формирование заключения."""
from __future__ import annotations

from typing import Dict, List

from .png_parser import PNGStructure

VERDICT_LABELS = {
    "CLEAN": "Скрытых вложений не выявлено",
    "SUSPICIOUS": "Подозрительное изображение",
    "STEGO_DETECTED": "Обнаружено скрытое вложение",
    "ERROR": "Ошибка анализа",
}
VERDICT_ORDER = {"CLEAN": 0, "SUSPICIOUS": 1, "STEGO_DETECTED": 2, "ERROR": 3}


def compute_verdict(stats: Dict, structure: PNGStructure, thr) -> Dict:
    stat_risk = 100.0 * stats.get("detection_probability", 0.0) if stats else 0.0
    struct_risk = structure.structural_risk
    risk = 100.0 * (1 - (1 - stat_risk / 100.0) * (1 - struct_risk / 100.0))
    reasons: List[str] = []
    recs: List[str] = []

    if stats:
        pe = stats["effective_payload"]
        if pe >= thr.payload_suspicious:
            reasons.append(f"Оценка доли LSB-вложения {pe:.1%} (порог {thr.payload_suspicious:.0%})")
        if stats["chi_square_max_p"] >= thr.chi_square_p:
            reasons.append(f"Атака хи-квадрат: p = {stats['chi_square_max_p']:.3f}, "
                           f"последовательное вложение ≈ {stats['sequential_length']:.0%} изображения")
    for f in structure.findings:
        if f.severity in ("medium", "high", "critical"):
            reasons.append(f"{f.title}: {f.details}" if f.details else f.title)

    codes = {f.code for f in structure.findings}
    if codes & {"TRAILING_DATA", "ZLIB_TRAILING", "IDAT_OVERSIZE"}:
        recs.append("Удалить данные вне пиксельной области: выполнить санитизацию (команда sanitize).")
    if codes & {"EMBEDDED_FILE"}:
        recs.append("Изолировать файл и передать на антивирусную проверку: обнаружены сигнатуры вложенных файлов.")
    if codes & {"UNKNOWN_CHUNK", "ENCODED_TEXT", "LONG_TEXT", "LARGE_ANCILLARY"}:
        recs.append("Удалить вспомогательные чанки при приёме файла (политика «только IHDR/PLTE/IDAT/IEND»).")
    if codes & {"CRC_MISMATCH", "CORRUPT_CHUNK", "TRUNCATED"}:
        recs.append("Проверить целостность источника: файл модифицирован или повреждён.")
    if stat_risk >= thr.risk_suspicious:
        recs.append("Перезаписать младшие биты (команда sanitize) перед передачей в промышленный сегмент.")
        recs.append("Выяснить источник изображения и сопоставить с эталоном (при наличии).")

    if risk >= thr.risk_stego:
        verdict = "STEGO_DETECTED"
    elif risk >= thr.risk_suspicious:
        verdict = "SUSPICIOUS"
    else:
        verdict = "CLEAN"
    if verdict == "CLEAN":
        recs.append("Действий не требуется. Для критичных контуров рекомендуется санитизация по умолчанию.")
    return {
        "verdict": verdict,
        "verdict_label": VERDICT_LABELS[verdict],
        "risk_score": round(risk, 1),
        "statistical_risk": round(stat_risk, 1),
        "structural_risk": round(struct_risk, 1),
        "reasons": reasons,
        "recommendations": recs,
    }
