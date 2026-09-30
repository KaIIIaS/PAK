from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None, quiet: bool = False) -> None:
    root = logging.getLogger("stegoguard")
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    if not quiet:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(fmt)
        root.addHandler(h)
    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
