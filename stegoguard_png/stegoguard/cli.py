"""Интерфейс командной строки StegoGuard PNG.

Коды возврата (для интеграции в SCADA/АСУ ТП, CI и шлюзы):
  0 - скрытых вложений не выявлено; 1 - подозрительно; 2 - обнаружено вложение; 3 - ошибка.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __app_name__, __version__
from .analyzer import Analyzer
from .config import Config
from .logging_setup import setup_logging
from .scoring import VERDICT_ORDER
from .utils import collect_files, human_bytes

EXIT = {"CLEAN": 0, "SUSPICIOUS": 1, "STEGO_DETECTED": 2, "ERROR": 3}


def _print_result(r: dict, verbose: bool = False) -> None:
    mark = {"CLEAN": "[ OK ]", "SUSPICIOUS": "[WARN]", "STEGO_DETECTED": "[STEG]", "ERROR": "[ERR ]"}[r["verdict"]]
    risk = "—" if r.get("risk_score") is None else f"{r['risk_score']:5.1f}"
    st = r.get("statistics") or {}
    pay = f"{st.get('effective_payload', 0):6.1%}" if st else "   —  "
    print(f"{mark} риск {risk}  доля {pay}  {r.get('file_name')}  — {r.get('verdict_label')}")
    if verbose or r["verdict"] != "CLEAN":
        for reason in r.get("reasons", [])[:8]:
            print(f"        • {reason}")
    rob = r.get("robustness")
    if rob and rob.get("supported"):
        print(f"        ИСС = {rob['stability_index']:.1f} ({rob['stability_class']}), безопасная ёмкость "
              f"{human_bytes(rob['max_secure_capacity_bytes'])} ({rob['most_dangerous_method']})")
    att = r.get("attacks")
    if att and att.get("supported"):
        print("        Нейтрализуют LSB-канал: " + (", ".join(att["neutralizing"]) or "нет"))


def cmd_analyze(args, cfg: Config) -> int:
    files = collect_files(args.paths, recursive=args.recursive, all_files=args.all_files)
    if not files:
        print("Файлы для анализа не найдены", file=sys.stderr)
        return 3
    analyzer = Analyzer(cfg)
    history = None
    if args.db:
        from .database import History
        history = History(args.db if isinstance(args.db, str) else cfg.db_path)
    results = []
    for i, f in enumerate(files, 1):
        r = analyzer.analyze_file(f, robustness=args.robustness, attacks=args.attacks)
        results.append(r)
        if history:
            history.add(r)
        if not args.json:
            _print_result(r, args.verbose)
    if args.report_dir:
        from .report import write_reports
        written = write_reports(results, args.report_dir, tuple(args.formats.split(",")))
        if not args.json:
            n = sum(len(v) for v in written.values())
            print(f"\nОтчёты ({n} файлов) сохранены в {os.path.abspath(args.report_dir)}")
    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2))
    elif len(results) > 1:
        counts = {}
        for r in results:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        print(f"\nИтого: {len(results)} файлов; " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    worst = max(results, key=lambda r: VERDICT_ORDER.get(r["verdict"], 3))["verdict"]
    return EXIT[worst]


def cmd_robustness(args, cfg: Config) -> int:
    if args.rates:
        cfg.robustness.rates = [float(x) for x in args.rates.split(",")]
    if args.trials:
        cfg.robustness.trials = args.trials
    cfg.validate()
    r = Analyzer(cfg).analyze_file(args.file, robustness=True, attacks=args.attacks)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        _print_result(r, True)
        rob = r.get("robustness") or {}
        if rob.get("supported"):
            for m in rob["methods"].values():
                print(f"\n  {m['label']}: ИСС {m['stability_index']:.1f} ({m['stability_class']}), безопасная доля "
                      f"{m['safe_rate']:.1%}, ёмкость {human_bytes(m['secure_capacity_bytes'])}")
                print("    доля     P_обн   оценка   PSNR     SSIM")
                for p in m["points"]:
                    print(f"    {p['rate']:6.1%}  {p['p_detect']:6.3f}  {p['payload_estimate']:6.1%}  {p['psnr']:6.2f}  {p['ssim']:.5f}")
        elif rob:
            print(rob.get("reason"))
    if args.report:
        from .report import save_html
        save_html(r, args.report, args.file)
        print(f"\nHTML-отчёт: {os.path.abspath(args.report)}")
    return EXIT[r["verdict"]]


def cmd_embed(args, cfg: Config) -> int:
    from .embedding import capacity_bytes, embed_message
    from .image_io import load_image
    img = load_image(args.cover)
    if args.message is not None:
        payload = args.message.encode("utf-8")
    else:
        with open(args.file, "rb") as fh:
            payload = fh.read()
    stego = embed_message(img, payload, key=args.key, method=args.method)
    stego.save_png(args.output)
    print(f"Встроено {len(payload)} байт из {capacity_bytes(img)} доступных "
          f"({len(payload) / max(capacity_bytes(img), 1):.1%}) методом {args.method} -> {args.output}")
    return 0


def cmd_extract(args, cfg: Config) -> int:
    from .embedding import ExtractionError, extract_message
    from .image_io import load_image
    try:
        data = extract_message(load_image(args.image), key=args.key)
    except ExtractionError as exc:
        print(f"Извлечение не удалось: {exc}", file=sys.stderr)
        return 3
    if args.output:
        with open(args.output, "wb") as fh:
            fh.write(data)
        print(f"Извлечено {len(data)} байт -> {args.output}")
    else:
        try:
            print(data.decode("utf-8"))
        except UnicodeDecodeError:
            print(f"Извлечено {len(data)} байт двоичных данных (используйте -o для сохранения)")
    return 0


def cmd_sanitize(args, cfg: Config) -> int:
    from .sanitizer import sanitize
    files = collect_files(args.inputs, recursive=args.recursive)
    multi = len(files) > 1 or os.path.isdir(args.output)
    for f in files:
        dst = os.path.join(args.output, os.path.basename(f)) if multi else args.output
        rep = sanitize(f, dst, lsb_mode=args.lsb_mode, include_alpha=args.alpha)
        print(f"{f} -> {dst}: удалены чанки {rep['removed_chunks'] or 'нет'}, после IEND {rep['removed_trailing_bytes']} байт, "
              f"LSB: {rep['lsb_mode']}, PSNR {rep['psnr']:.2f} дБ")
    return 0


def cmd_monitor(args, cfg: Config) -> int:
    from .database import History
    from .monitor import FolderMonitor
    history = History(args.db or cfg.db_path)
    mon = FolderMonitor(Analyzer(cfg), args.folder, args.interval, args.recursive, args.quarantine,
                        args.clean_dir, args.sanitize_dir, args.report_dir, history, args.quarantine_level)
    try:
        mon.run(max_cycles=args.cycles)
    except KeyboardInterrupt:
        print("\nОстановлено оператором")
    print(f"Обработано файлов: {mon.processed}")
    return 0


def cmd_history(args, cfg: Config) -> int:
    from .database import History
    h = History(args.db or cfg.db_path)
    if args.sha256:
        rows = h.by_hash(args.sha256)
    else:
        rows = h.recent(args.limit, args.verdict)
    for row in rows:
        print(f"#{row['id']:<5} {row['ts']}  {row['verdict']:<15} риск {row.get('risk')}  {row['file']}")
    print("Статистика: " + json.dumps(h.stats(), ensure_ascii=False))
    return 0


def cmd_samples(args, cfg: Config) -> int:
    from .samples import generate_dataset
    m = generate_dataset(args.output, count=args.count, seed=args.seed)
    print(f"Сформировано {len(m)} испытательных файлов в {os.path.abspath(args.output)} (разметка: manifest.json)")
    return 0


def cmd_benchmark(args, cfg: Config) -> int:
    from .benchmark import run_benchmark
    r = run_benchmark(args.dataset, Analyzer(cfg), not args.strict)
    print(f"Файлов: {r['total']}  TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")
    print(f"Точность (accuracy) {r['accuracy']:.1%}  TPR {r['tpr']:.1%}  FPR {r['fpr']:.1%}  Precision {r['precision']:.1%}")
    for k, v in r["by_type"].items():
        print(f"  {k:<40} {v['correct']}/{v['total']}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(r, fh, ensure_ascii=False, indent=2)
    return 0


def cmd_config(args, cfg: Config) -> int:
    cfg.save(args.output)
    print(f"Конфигурация по умолчанию сохранена в {args.output}")
    return 0


def cmd_gui(args, cfg: Config) -> int:
    from .gui import run_gui
    run_gui(cfg)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stegoguard", description=f"{__app_name__} {__version__}: анализ и оценка "
                                "стеганографической стойкости PNG-изображений в промышленных системах")
    p.add_argument("--version", action="version", version=f"{__app_name__} {__version__}")
    p.add_argument("-c", "--config", help="файл конфигурации JSON")
    p.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    p.add_argument("--log-file", default=None, help="журнал в файл (ротация 10 МБ)")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="анализ файлов/каталогов")
    a.add_argument("paths", nargs="+")
    a.add_argument("-r", "--recursive", action="store_true")
    a.add_argument("--all-files", action="store_true", help="анализировать файлы с любым расширением (выявление подмены типа)")
    a.add_argument("--robustness", action="store_true", help="оценить стеганографическую стойкость")
    a.add_argument("--attacks", action="store_true", help="оценить живучесть LSB-канала к преобразованиям")
    a.add_argument("-o", "--report-dir", help="каталог для отчётов")
    a.add_argument("--formats", default="json,html,csv")
    a.add_argument("--db", nargs="?", const=True, help="записать результаты в журнал SQLite")
    a.add_argument("--json", action="store_true", help="вывод результата в JSON")
    a.add_argument("-v", "--verbose", action="store_true")
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("robustness", help="оценка стеганографической стойкости изображения")
    r.add_argument("file")
    r.add_argument("--rates", help="доли заполнения через запятую, напр. 0.01,0.05,0.1")
    r.add_argument("--trials", type=int)
    r.add_argument("--attacks", action="store_true")
    r.add_argument("--report", help="путь HTML-отчёта")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_robustness)

    e = sub.add_parser("embed", help="эталонное встраивание (испытания, верификация)")
    e.add_argument("cover")
    e.add_argument("output")
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("-m", "--message")
    g.add_argument("-f", "--file")
    e.add_argument("-k", "--key", help="ключ псевдослучайного выбора позиций")
    e.add_argument("--method", choices=["lsb_replacement", "lsb_matching"], default="lsb_replacement")
    e.set_defaults(func=cmd_embed)

    x = sub.add_parser("extract", help="извлечение эталонного вложения")
    x.add_argument("image")
    x.add_argument("-k", "--key")
    x.add_argument("-o", "--output")
    x.set_defaults(func=cmd_extract)

    s = sub.add_parser("sanitize", help="нейтрализация скрытых каналов (CDR)")
    s.add_argument("inputs", nargs="+")
    s.add_argument("-o", "--output", required=True, help="файл или каталог назначения")
    s.add_argument("-r", "--recursive", action="store_true")
    s.add_argument("--lsb-mode", choices=["predictive", "random", "none"], default="predictive",
                   help="перезапись LSB: predictive - предсказательная (по умолчанию), random - случайная, none - не менять")
    s.add_argument("--alpha", action="store_true", help="обрабатывать также LSB альфа-канала")
    s.set_defaults(func=cmd_sanitize)

    m = sub.add_parser("monitor", help="непрерывный контроль каталога")
    m.add_argument("folder")
    m.add_argument("--interval", type=float, default=5.0)
    m.add_argument("-r", "--recursive", action="store_true")
    m.add_argument("--quarantine", help="каталог карантина")
    m.add_argument("--quarantine-level", choices=["SUSPICIOUS", "STEGO_DETECTED"], default="SUSPICIOUS")
    m.add_argument("--clean-dir", help="куда перемещать проверенные чистые файлы")
    m.add_argument("--sanitize-dir", help="куда сохранять санитизированные копии")
    m.add_argument("--report-dir")
    m.add_argument("--db")
    m.add_argument("--cycles", type=int, help="число циклов опроса (по умолчанию бесконечно)")
    m.set_defaults(func=cmd_monitor)

    h = sub.add_parser("history", help="журнал проверок")
    h.add_argument("--db")
    h.add_argument("--limit", type=int, default=20)
    h.add_argument("--verdict", choices=list(EXIT))
    h.add_argument("--sha256")
    h.set_defaults(func=cmd_history)

    sm = sub.add_parser("samples", help="сформировать испытательный набор")
    sm.add_argument("output")
    sm.add_argument("--count", type=int, default=3)
    sm.add_argument("--seed", type=int, default=7)
    sm.set_defaults(func=cmd_samples)

    b = sub.add_parser("benchmark", help="оценка точности на размеченном наборе")
    b.add_argument("dataset")
    b.add_argument("--strict", action="store_true", help="считать положительным только STEGO_DETECTED")
    b.add_argument("-o", "--output")
    b.set_defaults(func=cmd_benchmark)

    c = sub.add_parser("init-config", help="сохранить конфигурацию по умолчанию")
    c.add_argument("output")
    c.set_defaults(func=cmd_config)

    sub.add_parser("gui", help="графический интерфейс").set_defaults(func=cmd_gui)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = Config.load(args.config)
    except (OSError, ValueError) as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 3
    setup_logging(args.log_level or cfg.log_level, args.log_file or cfg.log_file,
                  quiet=getattr(args, "json", False) or (args.command in ("analyze", "robustness") and not args.log_level))
    try:
        return args.func(args, cfg)
    except FileNotFoundError as exc:
        print(f"Файл не найден: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
