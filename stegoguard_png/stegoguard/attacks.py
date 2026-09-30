"""Оценка живучести скрытого LSB-вложения к типовым преобразованиям изображения.

Позволяет определить, какие преобразования в контуре обработки изображений
гарантированно уничтожают скрытый канал (рекомендации по нейтрализации)."""
from __future__ import annotations

import io
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image, ImageFilter

from .config import Config
from .embedding import _embed_channels, embed_message, positions, read_bits, HEADER_LEN
from .image_io import ImageData, load_image


def _jpeg(q: int) -> Callable[[Image.Image], Image.Image]:
    def f(img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        src = img.convert("RGB") if img.mode not in ("L", "RGB") else img
        src.save(buf, format="JPEG", quality=q)
        return Image.open(io.BytesIO(buf.getvalue())).convert(img.mode)
    return f


def _png_resave(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return Image.open(io.BytesIO(buf.getvalue()))


def _noise(img):
    arr = np.asarray(img, dtype=np.float64)
    arr = arr + np.random.default_rng(0).normal(0, 1.0, arr.shape)
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), img.mode)


def _resize(img):
    w, h = img.size
    return img.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def _crop_pad(img):
    w, h = img.size
    dx, dy = max(1, w // 20), max(1, h // 20)
    out = Image.new(img.mode, (w, h))
    out.paste(img.crop((dx, dy, w, h)), (0, 0))
    return out


def _blur(img):
    return img.filter(ImageFilter.GaussianBlur(0.8))


def _quantize(img):
    return img.quantize(64).convert(img.mode) if img.mode in ("RGB", "L") else img


def _lsb_predictive(img):
    from .sanitizer import rewrite_lsb_predictive
    arr = np.asarray(img).astype(np.int64)
    rng = np.random.default_rng(1)
    if arr.ndim == 3:
        out = np.dstack([rewrite_lsb_predictive(arr[..., c], rng) for c in range(arr.shape[2])])
    else:
        out = rewrite_lsb_predictive(arr, rng)
    return Image.fromarray(out.astype(np.uint8), img.mode)


ATTACKS: Dict[str, tuple] = {
    "png_resave": ("Пересохранение PNG (без потерь)", _png_resave),
    "jpeg_95": ("JPEG, качество 95", _jpeg(95)),
    "jpeg_75": ("JPEG, качество 75", _jpeg(75)),
    "noise": ("Гауссов шум σ=1", _noise),
    "resize": ("Масштабирование 50% и обратно", _resize),
    "crop": ("Обрезка 5% с дополнением", _crop_pad),
    "blur": ("Размытие σ=0.8", _blur),
    "quantize": ("Квантование до 64 цветов", _quantize),
    "sanitize": ("Санитизация StegoGuard (перезапись LSB)", _lsb_predictive),
}


def assess_attacks(img: ImageData, cfg: Optional[Config] = None, rate: float = 0.1) -> Dict:
    cfg = cfg or Config()
    if img.is_palette or img.max_value != 255 or img.mode not in ("RGB", "RGBA", "L", "LA"):
        return {"supported": False, "reason": "Поддерживаются 8-битные RGB/RGBA/L/LA изображения."}
    work = img.crop_side(512)
    base = ImageData("RGB" if len([k for k in work.channels if k != "A"]) == 3 else "L",
                     work.width, work.height, {k: v for k, v in work.channels.items() if k != "A"},
                     255)
    _, flat = _embed_channels(base)
    key = f"attack-{cfg.seed}"
    rng = np.random.default_rng(cfg.seed)
    nbytes = max(16, int(flat.size * rate / 8) - HEADER_LEN)
    payload = rng.integers(0, 256, nbytes, dtype=np.uint8).tobytes()
    stego = embed_message(base, payload, key=key)
    pos = positions(flat.size, (nbytes + HEADER_LEN) * 8, key)
    ref = read_bits(stego, pos)
    pil = stego.to_pil()
    results = {}
    for code, (label, fn) in ATTACKS.items():
        try:
            attacked = load_image(fn(pil))
            if attacked.mode != base.mode:
                attacked = load_image(attacked.to_pil().convert(base.mode))
            got = read_bits(attacked, pos)
            ber = float(np.mean(got != ref)) if got.size == ref.size else 0.5
        except Exception as exc:  # noqa: BLE001
            results[code] = {"label": label, "error": str(exc)}
            continue
        results[code] = {"label": label, "ber": ber, "survives": ber < 0.01,
                         "destroyed": ber > 0.4,
                         "status": "сохраняется" if ber < 0.01 else ("разрушается" if ber > 0.4 else "частично повреждается")}
    return {"supported": True, "embedding_rate": rate, "results": results,
            "neutralizing": [v["label"] for v in results.values() if v.get("destroyed")]}
