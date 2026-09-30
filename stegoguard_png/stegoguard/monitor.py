"""Режим непрерывного контроля каталога (шлюз приёма изображений в промышленный сегмент)."""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from typing import Dict, Optional, Tuple

from .analyzer import Analyzer
from .database import History
from .report import save_html, save_json
from .sanitizer import sanitize
from .scoring import VERDICT_ORDER
from .utils import collect_files

log = logging.getLogger("stegoguard.monitor")


class FolderMonitor:
    def __init__(self, analyzer: Analyzer, folder: str, interval: float = 5.0, recursive: bool = False,
                 quarantine_dir: Optional[str] = None, clean_dir: Optional[str] = None,
                 sanitize_dir: Optional[str] = None, report_dir: Optional[str] = None,
                 history: Optional[History] = None, quarantine_level: str = "SUSPICIOUS"):
        self.analyzer = analyzer
        self.folder = folder
        self.interval = interval
        self.recursive = recursive
        self.quarantine_dir = quarantine_dir
        self.clean_dir = clean_dir
        self.sanitize_dir = sanitize_dir
        self.report_dir = report_dir
        self.history = history
        self.quarantine_level = quarantine_level
        self._seen: Dict[str, Tuple[float, int]] = {}
        self._pending: Dict[str, Tuple[float, int]] = {}
        self.processed = 0

    def _stable_new_files(self):
        ready = []
        for path in collect_files([self.folder], recursive=self.recursive):
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            sig = (st.st_mtime, st.st_size)
            if self._seen.get(path) == sig:
                continue
            if self._pending.get(path) == sig:     # размер не меняется между опросами -> запись завершена
                ready.append(path)
                self._pending.pop(path, None)
            else:
                self._pending[path] = sig
        return ready

    def process(self, path: str) -> Dict:
        result = self.analyzer.analyze_file(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        if self.report_dir:
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_json(result, os.path.join(self.report_dir, f"{stem}_{ts}.json"))
            if result["verdict"] != "CLEAN":
                save_html(result, os.path.join(self.report_dir, f"{stem}_{ts}.html"), path)
        if self.history:
            self.history.add(result)
        dangerous = VERDICT_ORDER.get(result["verdict"], 3) >= VERDICT_ORDER[self.quarantine_level]
        action = "оставлен"
        if dangerous and self.quarantine_dir:
            os.makedirs(self.quarantine_dir, exist_ok=True)
            dst = os.path.join(self.quarantine_dir, os.path.basename(path))
            shutil.move(path, dst)
            action = f"помещён в карантин: {dst}"
            path = dst
        if self.sanitize_dir and result["verdict"] != "ERROR" and (not dangerous or not self.quarantine_dir):
            dst = os.path.join(self.sanitize_dir, os.path.basename(path))
            sanitize(path, dst)
            action += f"; санитизированная копия: {dst}"
        elif self.clean_dir and not dangerous and os.path.exists(path):
            os.makedirs(self.clean_dir, exist_ok=True)
            dst = os.path.join(self.clean_dir, os.path.basename(path))
            shutil.move(path, dst)
            action = f"передан: {dst}"
        level = logging.WARNING if dangerous else logging.INFO
        log.log(level, "[%s] %s риск=%s -> %s", result["verdict"], os.path.basename(path), result.get("risk_score"), action)
        result["monitor_action"] = action
        self.processed += 1
        return result

    def run(self, stop_event: Optional[threading.Event] = None, max_cycles: Optional[int] = None) -> None:
        log.info("Контроль каталога %s (интервал %.1f с)", self.folder, self.interval)
        cycles = 0
        while not (stop_event and stop_event.is_set()):
            for path in self._stable_new_files():
                try:
                    st = os.stat(path)
                    self.process(path)
                    self._seen[path] = (st.st_mtime, st.st_size)
                except Exception:  # noqa: BLE001
                    log.exception("Ошибка обработки %s", path)
                    self._seen[path] = (0, -1)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            if stop_event:
                stop_event.wait(self.interval)
            else:
                time.sleep(self.interval)
