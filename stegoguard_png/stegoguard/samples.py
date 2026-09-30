"""Генерация синтетических испытательных контейнеров, характерных для промышленных систем
(снимки технологических камер, тепловизионные кадры, экраны HMI/SCADA, схемы), а также
модификаций со скрытыми вложениями различных типов. Используется для верификации комплекса."""
from __future__ import annotations

import io
import json
import os
import struct
import zlib
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .embedding import embed_message
from .image_io import load_image


def _smooth_field(rng, h, w, scales=(64, 16, 4), weights=(1.0, 0.5, 0.25)):
    acc = np.zeros((h, w))
    for s, wt in zip(scales, weights):
        small = rng.normal(size=(max(2, h // s + 2), max(2, w // s + 2)))
        img = Image.fromarray(((small - small.min()) / (np.ptp(small) + 1e-9) * 255).astype(np.uint8))
        acc += wt * (np.asarray(img.resize((w, h), Image.BICUBIC), dtype=np.float64) - 128)
    return acc


def cover_camera(rng, h=384, w=512) -> Image.Image:
    """Кадр технологической камеры: плавный фон, объекты с краями, шум сенсора."""
    base = 128 + 1.2 * _smooth_field(rng, h, w)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(img)
    for _ in range(12):
        x0, y0 = rng.integers(0, w - 40), rng.integers(0, h - 40)
        x1, y1 = x0 + rng.integers(20, 160), y0 + rng.integers(20, 120)
        col = tuple(int(c) for c in rng.integers(30, 230, 3))
        (d.rectangle if rng.random() < 0.5 else d.ellipse)([x0, y0, x1, y1], fill=col)
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    arr = np.asarray(img, dtype=np.float64)
    tint = rng.uniform(0.85, 1.15, 3)
    arr = arr * tint + rng.normal(0, 2.5, arr.shape)
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")


def cover_thermal(rng, h=384, w=512) -> Image.Image:
    """Тепловизионный кадр (оттенки серого) с «горячими точками» оборудования."""
    yy, xx = np.mgrid[0:h, 0:w]
    t = 70 + 0.6 * _smooth_field(rng, h, w, (96, 24), (1.0, 0.4))
    for _ in range(5):
        cx, cy, r = rng.integers(0, w), rng.integers(0, h), rng.integers(15, 60)
        t += rng.uniform(60, 140) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    t += rng.normal(0, 1.8, t.shape)
    return Image.fromarray(np.clip(np.rint(t), 0, 255).astype(np.uint8), "L")


def cover_hmi(rng, h=384, w=512) -> Image.Image:
    """Экран HMI/SCADA: плоские панели, индикаторы, текст, тренды."""
    img = Image.new("RGB", (w, h), (46, 52, 64))
    d = ImageDraw.Draw(img)
    for i in range(6):
        x0 = 10 + (i % 3) * (w // 3)
        y0 = 10 + (i // 3) * (h // 2)
        d.rectangle([x0, y0, x0 + w // 3 - 20, y0 + h // 2 - 20], fill=(59, 66, 82), outline=(136, 192, 208))
        d.text((x0 + 8, y0 + 6), f"PUMP-{i + 1:02d}  {rng.uniform(1, 9):.2f} bar", fill=(236, 239, 244))
        pts = [(x0 + 8 + k * 6, y0 + 90 + int(30 * np.sin(k / 4 + i) + rng.normal(0, 3))) for k in range(24)]
        d.line(pts, fill=(163, 190, 140), width=2)
        d.ellipse([x0 + 100, y0 + 30, x0 + 130, y0 + 60], fill=(191, 97, 106) if rng.random() < .3 else (163, 190, 140))
    return img


def cover_schematic(rng, h=384, w=512) -> Image.Image:
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        d.line([x0, y0, x0 + rng.integers(-120, 120), y0], fill=0, width=1)
        d.line([x0, y0, x0, y0 + rng.integers(-120, 120)], fill=0, width=1)
    return img.convert("P", palette=Image.ADAPTIVE, colors=16)


COVERS = {"camera": cover_camera, "thermal": cover_thermal, "hmi": cover_hmi, "schematic": cover_schematic}


# --- Структурные модификации PNG ------------------------------------------------------------

def _chunk(ctype: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def insert_chunk_before_iend(png: bytes, ctype: bytes, data: bytes) -> bytes:
    idx = png.rfind(b"IEND") - 4
    return png[:idx] + _chunk(ctype, data) + png[idx:]


def append_trailing(png: bytes, data: bytes) -> bytes:
    return png + data


def inject_idat_extra(png: bytes, extra: bytes) -> bytes:
    """Добавляет данные после конца zlib-потока внутри последнего IDAT."""
    pos, out, idats = 8, [png[:8]], []
    while pos < len(png):
        ln = struct.unpack(">I", png[pos:pos + 4])[0]
        ct = png[pos + 4:pos + 8]
        data = png[pos + 8:pos + 8 + ln]
        if ct == b"IDAT":
            idats.append(data)
        else:
            if ct == b"IEND" and idats:
                out.append(_chunk(b"IDAT", b"".join(idats) + extra))
                idats = []
            out.append(_chunk(ct, data))
        pos += 12 + ln
    return b"".join(out)


def corrupt_crc(png: bytes, ctype: bytes = b"IHDR") -> bytes:
    idx = png.find(ctype)
    ln = struct.unpack(">I", png[idx - 4:idx])[0]
    crc_pos = idx + 4 + ln
    b = bytearray(png)
    b[crc_pos] ^= 0xFF
    return bytes(b)


def generate_dataset(out_dir: str, count: int = 4, seed: int = 7, size=(384, 512)) -> List[Dict]:
    """Формирует набор испытательных файлов и manifest.json с эталонной разметкой."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    manifest: List[Dict] = []
    h, w = size

    def save(name: str, data: bytes, label: str, kind: str, detail: str):
        with open(os.path.join(out_dir, name), "wb") as fh:
            fh.write(data)
        manifest.append({"file": name, "label": label, "kind": kind, "detail": detail})

    for i in range(count):
        for kind, fn in COVERS.items():
            cover = fn(rng, h, w)
            base = f"{kind}_{i:02d}"
            clean = png_bytes(cover)
            save(f"{base}_clean.png", clean, "clean", kind, "исходный контейнер")
            if kind == "schematic":
                save(f"{base}_trailing.png", append_trailing(clean, b"PK\x03\x04" + os.urandom(2048)),
                     "stego", kind, "ZIP-данные после IEND")
                continue
            img = load_image(clean)
            cap = img.n_samples // 8
            for rate, method in ((0.1, "lsb_replacement"), (0.4, "lsb_replacement"),
                                 (1.0, "lsb_replacement"), (0.4, "lsb_matching")):
                payload = os.urandom(max(1, int(cap * rate) - 12))
                key = f"key{i}" if rate < 1.0 else None
                st = embed_message(img, payload, key=key, method=method)
                save(f"{base}_{method}_{int(rate * 100):03d}.png", png_bytes(st.to_pil()), "stego", kind,
                     f"{method}, заполнение {rate:.0%}{', последовательно' if key is None else ''}")
            if i == 0:
                save(f"{base}_trailing.png", append_trailing(clean, os.urandom(4096)), "stego", kind,
                     "случайные данные после IEND")
                save(f"{base}_ztxt.png", insert_chunk_before_iend(
                    clean, b"zTXt", b"Comment\x00\x00" + zlib.compress(os.urandom(3000).hex().encode())),
                    "stego", kind, "шестнадцатеричные данные в zTXt")
                save(f"{base}_private.png", insert_chunk_before_iend(clean, b"stEg", os.urandom(5000)),
                     "stego", kind, "частный чанк stEg")
                save(f"{base}_idat_extra.png", inject_idat_extra(clean, os.urandom(1500)), "stego", kind,
                     "данные после конца zlib-потока IDAT")
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest
