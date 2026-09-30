"""Графический интерфейс оператора (Tkinter)."""
from __future__ import annotations

import os
import queue
import threading
import webbrowser
from typing import Dict, List, Optional

from . import __app_name__, __version__
from .analyzer import Analyzer
from .config import Config
from .utils import collect_files, human_bytes

COLORS = {"CLEAN": "#dff0d8", "SUSPICIOUS": "#fcf8e3", "STEGO_DETECTED": "#f2dede", "ERROR": "#e0e0e0"}


def run_gui(cfg: Optional[Config] = None) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    cfg = cfg or Config()
    analyzer = Analyzer(cfg)
    root = tk.Tk()
    root.title(f"{__app_name__} {__version__}")
    root.geometry("1150x700")
    results: Dict[str, Dict] = {}
    files: List[str] = []
    q: "queue.Queue" = queue.Queue()
    busy = tk.BooleanVar(value=False)

    top = ttk.Frame(root, padding=6)
    top.pack(fill="x")
    rob_var = tk.BooleanVar(value=False)
    att_var = tk.BooleanVar(value=False)

    def add_files():
        sel = filedialog.askopenfilenames(filetypes=[("PNG", "*.png"), ("Все файлы", "*.*")])
        _add(list(sel))

    def add_folder():
        d = filedialog.askdirectory()
        if d:
            _add(collect_files([d], recursive=True))

    def _add(paths):
        for p in paths:
            if p not in files:
                files.append(p)
                tree.insert("", "end", iid=p, values=(os.path.basename(p), "ожидает", "", "", "", ""))
        status.set(f"Файлов в очереди: {len(files)}")

    def clear():
        files.clear()
        results.clear()
        tree.delete(*tree.get_children())
        details.delete("1.0", "end")

    def worker(paths, rob, att):
        for i, p in enumerate(paths, 1):
            q.put(("progress", p, i, len(paths)))
            r = analyzer.analyze_file(p, robustness=rob, attacks=att)
            q.put(("result", p, r))
        q.put(("done",))

    def start():
        if busy.get():
            return
        todo = [p for p in files if p not in results] or list(files)
        if not todo:
            messagebox.showinfo(__app_name__, "Добавьте файлы для анализа")
            return
        busy.set(True)
        pbar["maximum"] = len(todo)
        pbar["value"] = 0
        threading.Thread(target=worker, args=(todo, rob_var.get(), att_var.get()), daemon=True).start()

    def poll():
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == "progress":
                    status.set(f"Анализ {msg[2]}/{msg[3]}: {os.path.basename(msg[1])}")
                    tree.set(msg[1], "verdict", "анализ…")
                elif msg[0] == "result":
                    _, p, r = msg
                    results[p] = r
                    st = r.get("statistics") or {}
                    rob = r.get("robustness") or {}
                    tree.item(p, values=(r.get("file_name"), r.get("verdict_label"),
                                         "" if r.get("risk_score") is None else f"{r['risk_score']:.1f}",
                                         f"{st.get('effective_payload', 0):.1%}" if st else "—",
                                         len([f for f in (r.get("structure") or {}).get("findings", []) if f["severity"] != "info"]),
                                         f"{rob['stability_index']:.1f}" if rob.get("supported") else ""),
                              tags=(r["verdict"],))
                    pbar["value"] += 1
                elif msg[0] == "done":
                    busy.set(False)
                    counts = {}
                    for r in results.values():
                        counts[r["verdict_label"]] = counts.get(r["verdict_label"], 0) + 1
                    status.set("Готово. " + "; ".join(f"{k}: {v}" for k, v in counts.items()))
        except queue.Empty:
            pass
        root.after(150, poll)

    def show(_evt=None):
        sel = tree.selection()
        if not sel or sel[0] not in results:
            return
        r = results[sel[0]]
        details.delete("1.0", "end")
        lines = [f"Файл: {r.get('file')}", f"Заключение: {r.get('verdict_label')} (риск {r.get('risk_score')})",
                 f"SHA-256: {r.get('sha256')}", f"Размер: {human_bytes(r.get('file_size') or 0)}", ""]
        img = r.get("image")
        if img:
            lines.append(f"Изображение: {img['width']}x{img['height']}, {img['mode']}, {img['bit_depth']} бит")
        st = r.get("statistics") or {}
        if st:
            lines.append(f"Оценка доли вложения: {st['effective_payload']:.2%}; χ² p(max) = {st['chi_square_max_p']:.3f}")
            for name, c in st["channels"].items():
                if c.get("reliable"):
                    lines.append(f"  {name}: RS {c['rs']['estimate']:.3f}  SPA {c['spa']['estimate']:.3f}  WS {c['ws']['estimate']:.3f}  "
                                 f"итог {c['payload_estimate']:.3f}")
        lines.append("")
        lines.append("Основания:")
        lines += [f"  • {x}" for x in r.get("reasons") or ["признаков не выявлено"]]
        lines.append("Рекомендации:")
        lines += [f"  • {x}" for x in r.get("recommendations", [])]
        rob = r.get("robustness") or {}
        if rob.get("supported"):
            lines.append("")
            lines.append(f"ИСС: {rob['stability_index']:.1f} ({rob['stability_class']}), безопасная ёмкость "
                         f"{human_bytes(rob['max_secure_capacity_bytes'])}, наиболее опасный метод: {rob['most_dangerous_method']}")
        details.insert("1.0", "\n".join(lines))

    def save_report():
        sel = tree.selection()
        if not sel or sel[0] not in results:
            messagebox.showinfo(__app_name__, "Выберите проанализированный файл")
            return
        path = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML", "*.html"), ("JSON", "*.json")])
        if not path:
            return
        from .report import save_html, save_json
        (save_json if path.lower().endswith(".json") else save_html)(results[sel[0]], path)
        if path.lower().endswith(".html"):
            webbrowser.open("file://" + os.path.abspath(path))

    def save_all():
        if not results:
            return
        d = filedialog.askdirectory()
        if d:
            from .report import write_reports
            write_reports(list(results.values()), d)
            messagebox.showinfo(__app_name__, f"Отчёты сохранены в {d}")

    def do_sanitize():
        sel = tree.selection()
        if not sel:
            return
        dst = filedialog.asksaveasfilename(defaultextension=".png", initialfile="sanitized_" + os.path.basename(sel[0]))
        if dst:
            from .sanitizer import sanitize
            rep = sanitize(sel[0], dst)
            messagebox.showinfo(__app_name__, f"Сохранено: {dst}\nУдалены чанки: {rep['removed_chunks'] or 'нет'}\n"
                                              f"PSNR: {rep['psnr']:.2f} дБ")

    for text, cmd in (("Добавить файлы", add_files), ("Добавить папку", add_folder), ("Очистить", clear),
                      ("▶ Анализ", start), ("Отчёт…", save_report), ("Все отчёты…", save_all),
                      ("Санитизировать…", do_sanitize)):
        ttk.Button(top, text=text, command=cmd).pack(side="left", padx=2)
    ttk.Checkbutton(top, text="Оценка стойкости", variable=rob_var).pack(side="left", padx=8)
    ttk.Checkbutton(top, text="Атаки на канал", variable=att_var).pack(side="left")

    pane = ttk.PanedWindow(root, orient="vertical")
    pane.pack(fill="both", expand=True, padx=6)
    cols = ("file", "verdict", "risk", "payload", "findings", "iss")
    tree = ttk.Treeview(pane, columns=cols, show="headings", height=14)
    for c, t, w in zip(cols, ("Файл", "Заключение", "Риск", "Доля вложения", "Находки", "ИСС"),
                       (330, 260, 70, 110, 80, 70)):
        tree.heading(c, text=t)
        tree.column(c, width=w, anchor="w")
    for v, col in COLORS.items():
        tree.tag_configure(v, background=col)
    tree.bind("<<TreeviewSelect>>", show)
    pane.add(tree, weight=3)
    details = tk.Text(pane, height=14, font=("Consolas", 10), wrap="word")
    pane.add(details, weight=2)

    bottom = ttk.Frame(root, padding=4)
    bottom.pack(fill="x")
    status = tk.StringVar(value="Добавьте файлы или папку")
    ttk.Label(bottom, textvariable=status).pack(side="left")
    pbar = ttk.Progressbar(bottom, length=260)
    pbar.pack(side="right")
    root.after(150, poll)
    root.mainloop()
