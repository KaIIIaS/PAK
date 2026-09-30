from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Iterable, List

import numpy as np

PNG_EXT = {".png", ".apng"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    return obj


def collect_files(paths: Iterable[str], recursive: bool = False, all_files: bool = False) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            walker = os.walk(p) if recursive else [(p, [], os.listdir(p))]
            for root, _dirs, files in walker:
                for f in sorted(files):
                    full = os.path.join(root, f)
                    if os.path.isfile(full) and (all_files or os.path.splitext(f)[1].lower() in PNG_EXT):
                        out.append(full)
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


def human_bytes(n: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ТБ"
