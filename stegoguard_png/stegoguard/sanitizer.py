"""Нейтрализация (санитизация) PNG: удаление вспомогательных чанков, данных после IEND
и потенциального LSB-канала без заметного изменения изображения (CDR-подход)."""
from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional

import numpy as np
from PIL import Image

from .metrics import psnr
from .png_parser import parse_png

KEEP_CHUNKS = {"IHDR", "PLTE", "IDAT", "IEND", "tRNS"}
LSB_MODES = ("predictive", "random", "none")


def rewrite_lsb_predictive(ch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Предсказательная перезапись LSB. Новый младший бит каждого отсчёта выбирается так, чтобы
    значение было ближайшим к предсказанию по четырём соседям, вычисленному ТОЛЬКО по старшим
    битам (x & ~1) + 0.5 (при равенстве бит определяется старшими битами). Результат не зависит от исходных LSB (скрытый канал уничтожается
    полностью), а статистика изображения остаётся естественной: в отличие от случайной замены
    LSB, такой образ не вызывает ложных срабатываний стегодетекторов."""
    hi = ch.astype(np.int64) & ~1
    base = np.pad(hi.astype(np.float64) + 0.5, 1, mode="edge")
    pred = (base[:-2, 1:-1] + base[2:, 1:-1] + base[1:-1, :-2] + base[1:-1, 2:]) / 4.0
    diff = pred - (hi + 0.5)
    # при равенстве - детерминированно по старшим битам: однородные области остаются однородными
    lsb = np.where(diff > 1e-9, 1, np.where(diff < -1e-9, 0, (hi >> 1) & 1))
    return hi | lsb


def rewrite_lsb_random(ch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (ch.astype(np.int64) & ~1) | rng.integers(0, 2, ch.shape)


def sanitize(src: str, dst: str, lsb_mode: str = "predictive", include_alpha: bool = False,
             seed: Optional[int] = None) -> Dict:
    """lsb_mode: predictive (по умолчанию) | random | none."""
    if lsb_mode not in LSB_MODES:
        raise ValueError(f"lsb_mode должен быть одним из {LSB_MODES}")
    with open(src, "rb") as fh:
        raw = fh.read()
    structure = parse_png(raw) if raw.startswith(b"\x89PNG") else None
    img = Image.open(src)
    img.load()
    mode = img.mode
    notes = []
    if mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        notes.append("Палитровое изображение преобразовано в полноцветное.")
    elif mode in ("1", "CMYK", "YCbCr", "LAB", "HSV", "PA", "La", "RGBa"):
        img = img.convert("RGBA" if "A" in mode or "a" in mode else "RGB")
    mode = img.mode

    arr = np.asarray(img)
    original = arr.copy()
    if lsb_mode != "none":
        fn = rewrite_lsb_predictive if lsb_mode == "predictive" else rewrite_lsb_random
        rng = np.random.default_rng(seed if seed is not None else int.from_bytes(os.urandom(8), "big"))
        work = arr.astype(np.int64).copy()
        if work.ndim == 3:
            n_color = work.shape[2] - (1 if mode in ("RGBA", "LA") else 0)
            idx = list(range(n_color)) + ([work.shape[2] - 1] if include_alpha and n_color < work.shape[2] else [])
            for c in idx:
                work[..., c] = fn(work[..., c], rng)
        else:
            work = fn(work, rng)
        arr = work.astype(original.dtype)
        img = Image.fromarray(arr, mode if mode != "I;16" else None)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    img.save(dst, format="PNG")  # без pnginfo: вспомогательные чанки не переносятся

    with open(dst, "rb") as fh:
        out = fh.read()
    removed = sorted({c.type for c in structure.chunks} - KEEP_CHUNKS) if structure else []
    maxv = 65535 if (original.dtype == np.uint16 or mode.startswith("I")) else 255
    return {
        "input": src, "output": dst,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "output_sha256": hashlib.sha256(out).hexdigest(),
        "input_size": len(raw), "output_size": len(out),
        "removed_chunks": removed,
        "removed_trailing_bytes": structure.trailing_bytes if structure else 0,
        "lsb_mode": lsb_mode,
        "psnr": psnr(original, arr, maxv),
        "notes": notes,
    }
