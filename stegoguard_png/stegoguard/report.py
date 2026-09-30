"""Формирование отчётов: JSON, CSV (сводный), HTML (с графиками)."""
from __future__ import annotations

import base64
import csv
import html
import io
import json
import os
from typing import Dict, List, Optional

import numpy as np

from .utils import human_bytes

VERDICT_COLORS = {"CLEAN": "#2e7d32", "SUSPICIOUS": "#ef6c00", "STEGO_DETECTED": "#c62828", "ERROR": "#616161"}
SEV_COLORS = {"info": "#607d8b", "low": "#689f38", "medium": "#f9a825", "high": "#ef6c00", "critical": "#c62828"}


def save_json(result: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return path


CSV_FIELDS = ["file_name", "verdict", "verdict_label", "risk_score", "statistical_risk", "structural_risk",
              "effective_payload", "chi_square_max_p", "findings", "stability_index", "secure_capacity_bytes",
              "width", "height", "mode", "file_size", "sha256", "file", "elapsed_sec"]


def csv_row(r: Dict) -> Dict:
    st = r.get("statistics") or {}
    img = r.get("image") or {}
    rob = r.get("robustness") or {}
    findings = [f["code"] for f in (r.get("structure") or {}).get("findings", []) if f["severity"] != "info"]
    return {
        "file_name": r.get("file_name"), "verdict": r.get("verdict"), "verdict_label": r.get("verdict_label"),
        "risk_score": r.get("risk_score"), "statistical_risk": r.get("statistical_risk"),
        "structural_risk": r.get("structural_risk"),
        "effective_payload": round(st.get("effective_payload", 0.0), 4) if st else "",
        "chi_square_max_p": round(st.get("chi_square_max_p", 0.0), 4) if st else "",
        "findings": ";".join(findings),
        "stability_index": round(rob["stability_index"], 1) if rob.get("supported") else "",
        "secure_capacity_bytes": rob.get("max_secure_capacity_bytes", "") if rob.get("supported") else "",
        "width": img.get("width"), "height": img.get("height"), "mode": img.get("mode"),
        "file_size": r.get("file_size"), "sha256": r.get("sha256"), "file": r.get("file"),
        "elapsed_sec": r.get("elapsed_sec"),
    }


def save_csv(results: List[Dict], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, delimiter=";")
        w.writeheader()
        for r in results:
            w.writerow(csv_row(r))
    return path


# ---------------------------------------------------------------- графики

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _charts(result: Dict, image_path: Optional[str]) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts: Dict[str, str] = {}
    colors = {"R": "#d32f2f", "G": "#388e3c", "B": "#1976d2", "Y": "#424242", "index": "#6a1b9a", "A": "#9e9e9e"}
    img = None
    if image_path and os.path.exists(image_path):
        try:
            from .image_io import load_image
            img = load_image(image_path).crop_side(768)
        except Exception:  # noqa: BLE001
            img = None
    if img is not None:
        fig, ax = plt.subplots(figsize=(7, 2.6))
        for name, ch in img.channels.items():
            h = np.bincount(ch.ravel(), minlength=img.max_value + 1)
            if img.max_value > 255:
                h = h.reshape(256, -1).sum(1)
            ax.plot(h, color=colors.get(name, "k"), lw=0.8, label=name)
        ax.set_title("Гистограммы каналов")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 255)
        charts["hist"] = _fig_to_b64(fig)

        names = list(img.channels)[:3]
        fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3))
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, names):
            ax.imshow(img.channels[name] & 1, cmap="gray", interpolation="nearest")
            ax.set_title(f"LSB-плоскость: {name}", fontsize=9)
            ax.axis("off")
        charts["lsb"] = _fig_to_b64(fig)

    st = result.get("statistics") or {}
    chans = {k: v for k, v in (st.get("channels") or {}).items() if v.get("progressive_chi")}
    if chans:
        fig, ax = plt.subplots(figsize=(7, 2.6))
        for name, v in chans.items():
            xs = [p["fraction"] * 100 for p in v["progressive_chi"]]
            ys = [p["p_value"] for p in v["progressive_chi"]]
            ax.plot(xs, ys, marker=".", color=colors.get(name, "k"), label=name)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Доля изображения, %")
        ax.set_ylabel("p")
        ax.set_title("Атака хи-квадрат (последовательное вложение)")
        ax.legend(fontsize=7)
        charts["chi"] = _fig_to_b64(fig)

    rob = result.get("robustness") or {}
    if rob.get("supported"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        for code, m in rob["methods"].items():
            r = [p["rate"] for p in m["points"]]
            axes[0].semilogx(r, [p["p_detect"] for p in m["points"]], marker="o", label=m["label"])
            axes[1].semilogx(r, [p["psnr"] for p in m["points"]], marker="o", label=m["label"])
        axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)
        axes[0].set_title("Вероятность обнаружения")
        axes[0].set_xlabel("Доля заполнения, бит/отсчёт")
        axes[0].set_ylim(-0.05, 1.05)
        axes[1].set_title("PSNR стегообраза, дБ")
        axes[1].set_xlabel("Доля заполнения, бит/отсчёт")
        for ax in axes:
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        charts["rob"] = _fig_to_b64(fig)

    att = result.get("attacks") or {}
    if att.get("supported"):
        items = [(v["label"], v.get("ber", 0.5)) for v in att["results"].values()]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh([i[0] for i in items], [i[1] for i in items],
                color=["#2e7d32" if b > 0.4 else ("#c62828" if b < 0.01 else "#f9a825") for _, b in items])
        ax.set_xlim(0, 0.55)
        ax.set_xlabel("BER извлечения вложения")
        ax.set_title("Живучесть LSB-канала к преобразованиям")
        charts["att"] = _fig_to_b64(fig)
    return charts


# ---------------------------------------------------------------- HTML

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#1f2933}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;border-bottom:2px solid #cfd8dc;padding-bottom:4px;margin-top:28px}
.card{background:#fff;border-radius:8px;padding:16px 20px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:16px}
.verdict{color:#fff;border-radius:8px;padding:16px 20px;display:flex;justify-content:space-between;align-items:center}
.verdict .big{font-size:22px;font-weight:600} .risk{font-size:34px;font-weight:700}
table{border-collapse:collapse;width:100%;font-size:13px} th,td{border-bottom:1px solid #e0e6ea;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#eceff1} .sev{color:#fff;border-radius:4px;padding:1px 6px;font-size:11px;white-space:nowrap}
.muted{color:#607d8b;font-size:12px} img.chart{max-width:100%} .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.bar{height:10px;background:#e0e6ea;border-radius:5px;overflow:hidden}.bar>div{height:100%}
ul{margin:6px 0 0 18px;padding:0} code{background:#eceff1;padding:1px 4px;border-radius:3px}
"""


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _pct(x) -> str:
    try:
        return f"{float(x):.1%}"
    except (TypeError, ValueError):
        return "—"


def render_html(result: Dict, image_path: Optional[str] = None) -> str:
    charts = _charts(result, image_path or result.get("file"))
    v = result.get("verdict", "ERROR")
    color = VERDICT_COLORS.get(v, "#616161")
    risk = result.get("risk_score")
    img = result.get("image") or {}
    st = result.get("statistics") or {}
    structure = result.get("structure") or {}
    parts = [f"<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'><title>Отчёт StegoGuard: {_e(result.get('file_name'))}</title><style>{CSS}</style></head><body><div class='wrap'>"]
    parts.append(f"<h1>Отчёт о стеганографическом анализе PNG</h1><div class='muted'>StegoGuard PNG v{_e(result.get('version'))} · {_e(result.get('timestamp'))} · время анализа {_e(result.get('elapsed_sec'))} с</div>")
    parts.append(f"<div class='verdict' style='background:{color};margin-top:14px'><div><div class='big'>{_e(result.get('verdict_label'))}</div>"
                 f"<div>{_e(result.get('file_name'))}</div></div><div class='risk'>{'—' if risk is None else f'{risk:.0f}'}<span style='font-size:14px'>/100</span></div></div>")

    # Сводка
    parts.append("<h2>Сведения об объекте</h2><div class='card'><table>")
    rows = [("Файл", result.get("file")), ("Размер", human_bytes(result.get("file_size") or 0)),
            ("SHA-256", result.get("sha256")),
            ("Изображение", f"{img.get('width')}×{img.get('height')}, режим {img.get('mode')}, {img.get('bit_depth')} бит/канал" if img else result.get("image_error")),
            ("Статистический риск", result.get("statistical_risk")), ("Структурный риск", result.get("structural_risk"))]
    if img.get("analyzed_fragment"):
        rows.append(("Анализируемый фрагмент", img["analyzed_fragment"]))
    parts += [f"<tr><th style='width:220px'>{_e(k)}</th><td>{_e(val)}</td></tr>" for k, val in rows]
    parts.append("</table></div>")

    if result.get("reasons") or result.get("recommendations"):
        parts.append("<h2>Основания и рекомендации</h2><div class='card grid'><div><b>Основания заключения</b><ul>")
        parts += [f"<li>{_e(x)}</li>" for x in result.get("reasons") or ["Признаков скрытых вложений не выявлено"]]
        parts.append("</ul></div><div><b>Рекомендации</b><ul>")
        parts += [f"<li>{_e(x)}</li>" for x in result.get("recommendations") or []]
        parts.append("</ul></div></div>")

    # Структура
    if structure:
        parts.append("<h2>Структурный анализ PNG</h2><div class='card'>")
        findings = structure.get("findings", [])
        if findings:
            parts.append("<table><tr><th>Критичность</th><th>Находка</th><th>Подробности</th><th>Смещение</th></tr>")
            for f in findings:
                parts.append(f"<tr><td><span class='sev' style='background:{SEV_COLORS[f['severity']]}'>{_e(f['severity_label'])}</span></td>"
                             f"<td>{_e(f['title'])}</td><td>{_e(f['details'])}</td><td>{_e(f.get('offset'))}</td></tr>")
            parts.append("</table>")
        else:
            parts.append("<p>Структурных аномалий не обнаружено.</p>")
        parts.append("<p class='muted'>Чанки: " + ", ".join(
            f"{_e(k)}×{s['count']} ({human_bytes(s['bytes'])})" for k, s in structure.get("chunk_summary", {}).items())
            + f". Данных после IEND: {structure.get('trailing_bytes', 0)} байт. IDAT: сжато {human_bytes(structure.get('idat_total') or 0)}, "
              f"распаковано {structure.get('decompressed_size')} из ожидаемых {structure.get('expected_size')} байт.</p>")
        if structure.get("text_entries"):
            parts.append("<table><tr><th>Чанк</th><th>Ключ</th><th>Длина</th><th>Энтропия</th><th>Фрагмент</th></tr>")
            for t in structure["text_entries"]:
                parts.append(f"<tr><td>{_e(t['chunk'])}</td><td>{_e(t['keyword'])}</td><td>{t['length']}</td><td>{t['entropy']}</td><td><code>{_e(t['preview'][:80])}</code></td></tr>")
            parts.append("</table>")
        parts.append("</div>")

    # Статистика
    if st:
        parts.append("<h2>Статистический стегоанализ</h2><div class='card'>")
        parts.append(f"<p>Интегральная оценка доли вложения: <b>{_pct(st.get('effective_payload'))}</b> "
                     f"(среднее по каналам {_pct(st.get('payload_estimate'))}, максимум {_pct(st.get('payload_max_channel'))}); "
                     f"вероятность наличия вложения: <b>{_pct(st.get('detection_probability'))}</b>.</p>")
        parts.append("<table><tr><th>Канал</th><th>RS</th><th>SPA</th><th>WS</th><th>FRN</th><th>χ² p</th><th>Оценка</th><th>LSB: доля 1</th><th>Энтропия LSB</th><th>Шум σ</th></tr>")
        for name, c in st.get("channels", {}).items():
            if not c.get("reliable"):
                parts.append(f"<tr><td>{_e(name)}</td><td colspan='9' class='muted'>не анализировался: {_e(c.get('reason'))}</td></tr>")
                continue
            frn = c.get("frn", {})
            parts.append(f"<tr><td><b>{_e(name)}</b></td><td>{_pct(c['rs']['estimate'])}</td><td>{_pct(c['spa']['estimate'])}</td>"
                         f"<td>{_pct(c['ws']['estimate'])}</td><td>{_pct(frn['estimate']) if frn.get('applicable') else '—'}</td>"
                         f"<td>{c['chi_square']['p_value']:.3f}</td><td><b>{_pct(c['payload_estimate'])}</b></td>"
                         f"<td>{c['lsb']['ones_ratio']:.3f}</td><td>{c['lsb']['block_entropy']:.3f}</td><td>{c['noise_sigma']:.2f}</td></tr>")
        parts.append("</table>")
        for n in st.get("notes", []):
            parts.append(f"<p class='muted'>{_e(n)}</p>")
        for key in ("hist", "lsb", "chi"):
            if key in charts:
                parts.append(f"<img class='chart' src='data:image/png;base64,{charts[key]}'>")
        parts.append("</div>")

    cx = result.get("complexity")
    if cx:
        parts.append(f"<div class='card'><b>Характеристики контейнера:</b> шум σ={cx['noise_sigma']:.2f}, средний градиент {cx['mean_gradient']:.2f}, "
                     f"насыщенные отсчёты {_pct(cx['saturation_ratio'])}, сложность текстуры — <b>{_e(cx['complexity_label'])}</b> "
                     f"({cx['complexity_index']:.2f}).</div>")

    rob = result.get("robustness")
    if rob:
        parts.append("<h2>Оценка стеганографической стойкости</h2><div class='card'>")
        if not rob.get("supported"):
            parts.append(f"<p>{_e(rob.get('reason'))}</p>")
        else:
            idx = rob["stability_index"]
            parts.append(f"<p>Индекс стеганографической стойкости (ИСС): <b style='font-size:20px'>{idx:.1f}</b> / 100 — "
                         f"<b>{_e(rob['stability_class'])}</b>. Наиболее опасный метод: {_e(rob['most_dangerous_method'])}. "
                         f"Максимальная незаметная для детекторов ёмкость: <b>{human_bytes(rob['max_secure_capacity_bytes'])}</b> "
                         f"из {human_bytes(rob['total_capacity_bytes'])} полной LSB-ёмкости.</p>"
                         f"<div class='bar'><div style='width:{idx:.0f}%;background:{'#c62828' if idx >= 70 else ('#ef6c00' if idx >= 40 else '#2e7d32')}'></div></div>"
                         f"<p class='muted'>Высокий ИСС означает, что изображение является «удобным» контейнером: значительный объём данных может "
                         f"быть скрыт без обнаружения средствами контроля. Испытания проводились на фрагменте {rob['fragment']}.</p>")
            parts.append("<table><tr><th>Метод</th><th>ИСС</th><th>Класс</th><th>Безопасная доля</th><th>Первое обнаружение</th><th>Безопасная ёмкость</th></tr>")
            for m in rob["methods"].values():
                fd = m["first_detected_rate"]
                parts.append(f"<tr><td>{_e(m['label'])}</td><td>{m['stability_index']:.1f}</td><td>{_e(m['stability_class'])}</td>"
                             f"<td>{_pct(m['safe_rate'])}</td><td>{'не обнаруживается' if fd is None else _pct(fd)}</td>"
                             f"<td>{human_bytes(m['secure_capacity_bytes'])}</td></tr>")
            parts.append("</table><table style='margin-top:10px'><tr><th>Метод</th><th>Доля r</th><th>P обнаружения</th><th>Оценка доли</th><th>PSNR, дБ</th><th>SSIM</th></tr>")
            for m in rob["methods"].values():
                for p in m["points"]:
                    parts.append(f"<tr><td>{_e(m['label'])}</td><td>{_pct(p['rate'])}</td><td>{p['p_detect']:.3f}</td><td>{_pct(p['payload_estimate'])}</td>"
                                 f"<td>{p['psnr']:.2f}</td><td>{p['ssim']:.5f}</td></tr>")
            parts.append("</table>")
            for n in rob.get("notes", []):
                parts.append(f"<p class='muted'>{_e(n)}</p>")
            if "rob" in charts:
                parts.append(f"<img class='chart' src='data:image/png;base64,{charts['rob']}'>")
        parts.append("</div>")

    att = result.get("attacks")
    if att:
        parts.append("<h2>Живучесть скрытого канала к преобразованиям</h2><div class='card'>")
        if not att.get("supported"):
            parts.append(f"<p>{_e(att.get('reason'))}</p>")
        else:
            parts.append("<table><tr><th>Преобразование</th><th>BER</th><th>Состояние LSB-канала</th></tr>")
            for a in att["results"].values():
                parts.append(f"<tr><td>{_e(a['label'])}</td><td>{a.get('ber', float('nan')):.4f}</td><td>{_e(a.get('status', a.get('error')))}</td></tr>")
            parts.append("</table>")
            parts.append("<p>Преобразования, гарантированно нейтрализующие LSB-канал: " + _e(", ".join(att["neutralizing"]) or "нет") + ".</p>")
            if "att" in charts:
                parts.append(f"<img class='chart' src='data:image/png;base64,{charts['att']}'>")
        parts.append("</div>")

    parts.append("<p class='muted'>Методы: атака хи-квадрат (Westfeld–Pfitzmann), RS-анализ (Fridrich–Goljan–Du), SPA (Dumitrescu–Wu–Wang), "
                 "взвешенный стегоанализ WS (Fridrich–Goljan, Ker–Böhme), тест однородных областей FRN, структурный анализ по ISO/IEC 15948.</p>")
    parts.append("</div></body></html>")
    return "".join(parts)


def save_html(result: Dict, path: str, image_path: Optional[str] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(result, image_path))
    return path


def render_summary_html(results: List[Dict], links: Optional[Dict[str, str]] = None) -> str:
    links = links or {}
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.get("verdict", "ERROR")] = counts.get(r.get("verdict", "ERROR"), 0) + 1
    parts = [f"<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'><title>Сводный отчёт StegoGuard</title><style>{CSS}</style></head><body><div class='wrap'>",
             f"<h1>Сводный отчёт о проверке PNG-изображений</h1><div class='muted'>Проверено файлов: {len(results)}</div><div class='card' style='margin-top:12px'>"]
    from .scoring import VERDICT_LABELS
    parts.append(" · ".join(f"<b style='color:{VERDICT_COLORS[k]}'>{_e(VERDICT_LABELS[k])}: {counts.get(k, 0)}</b>" for k in VERDICT_LABELS))
    parts.append("</div><div class='card'><table><tr><th>Файл</th><th>Заключение</th><th>Риск</th><th>Оценка доли</th><th>Находки</th><th>ИСС</th></tr>")
    for r in sorted(results, key=lambda x: -(x.get("risk_score") or 0)):
        row = csv_row(r)
        name = _e(r.get("file_name"))
        link = links.get(r.get("file"))
        if link:
            name = f"<a href='{_e(link)}'>{name}</a>"
        parts.append(f"<tr><td>{name}</td><td style='color:{VERDICT_COLORS.get(r.get('verdict'), '#000')}'><b>{_e(r.get('verdict_label'))}</b></td>"
                     f"<td>{_e(r.get('risk_score'))}</td><td>{_pct(row['effective_payload']) if row['effective_payload'] != '' else '—'}</td>"
                     f"<td>{_e(row['findings'].replace(';', ', '))}</td><td>{_e(row['stability_index'])}</td></tr>")
    parts.append("</table></div></div></body></html>")
    return "".join(parts)


def write_reports(results: List[Dict], out_dir: str, formats=("json", "html", "csv")) -> Dict[str, List[str]]:
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, List[str]] = {"json": [], "html": [], "csv": []}
    links = {}
    used = set()
    for r in results:
        stem = os.path.splitext(r.get("file_name") or "report")[0]
        base, k = stem, 1
        while stem in used:
            k += 1
            stem = f"{base}_{k}"
        used.add(stem)
        if "json" in formats:
            written["json"].append(save_json(r, os.path.join(out_dir, f"{stem}.json")))
        if "html" in formats:
            p = save_html(r, os.path.join(out_dir, f"{stem}.html"))
            written["html"].append(p)
            links[r.get("file")] = os.path.basename(p)
    if "csv" in formats:
        written["csv"].append(save_csv(results, os.path.join(out_dir, "summary.csv")))
    if "html" in formats and len(results) > 1:
        p = os.path.join(out_dir, "index.html")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(render_summary_html(results, links))
        written["html"].append(p)
    return written
