"""Эталонные LSB-методы встраивания для испытаний стойкости и верификации детекторов.

Формат вложения: MAGIC(4) | длина(4, BE) | CRC32(4, BE) | данные.
При заданном ключе позиции отсчётов выбираются псевдослучайной перестановкой,
порождённой SHA-256(ключ); без ключа - последовательно.
"""
from __future__ import annotations

import hashlib
import struct
import zlib
from typing import Dict, Optional, Tuple

import numpy as np

from .image_io import ImageData

MAGIC = b"SGP1"
HEADER_LEN = 12
METHODS = ("lsb_replacement", "lsb_matching")


class CapacityError(ValueError):
    pass


class ExtractionError(ValueError):
    pass


def key_rng(key: Optional[str], salt: bytes = b"positions") -> np.random.Generator:
    digest = hashlib.sha256(salt + (key or "").encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def apply_bits(values: np.ndarray, bits: np.ndarray, method: str, rng: np.random.Generator,
               max_value: int = 255) -> np.ndarray:
    v = values.astype(np.int64).copy()
    bits = bits.astype(np.int64)
    if method == "lsb_replacement":
        return (v & ~1) | bits
    if method == "lsb_matching":
        need = (v & 1) != bits
        step = rng.choice(np.array([-1, 1]), size=v.size)
        step = np.where(v == 0, 1, np.where(v == max_value, -1, step))
        v[need] += step[need]
        return v
    raise ValueError(f"Неизвестный метод встраивания: {method}")


def _embed_channels(img: ImageData):
    if img.is_palette:
        raise ValueError("Встраивание в палитровые изображения не поддерживается")
    names = [n for n in img.channels if n != "A"]
    return names, np.concatenate([img.channels[n].ravel() for n in names])


def _restore(img: ImageData, names, flat) -> ImageData:
    out, pos = {}, 0
    for n in names:
        size = img.channels[n].size
        out[n] = flat[pos:pos + size].reshape(img.channels[n].shape)
        pos += size
    for n, ch in img.channels.items():
        out.setdefault(n, ch.copy())
    return img.copy_with(out)


def capacity_bytes(img: ImageData) -> int:
    names, flat = _embed_channels(img)
    return max(flat.size // 8 - HEADER_LEN, 0)


def positions(n: int, count: int, key: Optional[str]) -> np.ndarray:
    if key:
        return key_rng(key).permutation(n)[:count]
    return np.arange(count)


def embed_message(img: ImageData, payload: bytes, key: Optional[str] = None,
                  method: str = "lsb_replacement") -> ImageData:
    names, flat = _embed_channels(img)
    blob = MAGIC + struct.pack(">II", len(payload), zlib.crc32(payload) & 0xFFFFFFFF) + payload
    bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8)).astype(np.int64)
    if bits.size > flat.size:
        raise CapacityError(f"Вложение {len(payload)} байт превышает ёмкость {capacity_bytes(img)} байт")
    pos = positions(flat.size, bits.size, key)
    flat = flat.copy()
    flat[pos] = apply_bits(flat[pos], bits, method, key_rng(key, b"matching"), img.max_value)
    return _restore(img, names, flat)


def extract_message(img: ImageData, key: Optional[str] = None) -> bytes:
    names, flat = _embed_channels(img)
    n = flat.size
    perm = key_rng(key).permutation(n) if key else None

    def read(start: int, count: int) -> bytes:
        idx = perm[start:start + count] if perm is not None else np.arange(start, start + count)
        if idx.size < count:
            raise ExtractionError("Вложение выходит за пределы контейнера")
        return np.packbits((flat[idx] & 1).astype(np.uint8)).tobytes()

    header = read(0, HEADER_LEN * 8)
    if header[:4] != MAGIC:
        raise ExtractionError("Вложение не найдено (неверный ключ или контейнер без вложения)")
    length, crc = struct.unpack(">II", header[4:12])
    if length * 8 > n - HEADER_LEN * 8:
        raise ExtractionError("Повреждённый заголовок вложения")
    data = read(HEADER_LEN * 8, length * 8)
    if zlib.crc32(data) & 0xFFFFFFFF != crc:
        raise ExtractionError("Контрольная сумма вложения не совпадает (данные повреждены)")
    return data


def read_bits(img: ImageData, pos: np.ndarray) -> np.ndarray:
    names, flat = _embed_channels(img)
    pos = pos[pos < flat.size]
    return (flat[pos] & 1).astype(np.int64)


def embed_random_rate(channels: Dict[str, np.ndarray], rate: float, method: str,
                      rng: np.random.Generator, max_value: int = 255) -> Dict[str, np.ndarray]:
    """Встраивание случайного сообщения с долей заполнения rate (бит на отсчёт) в случайные позиции."""
    out = {}
    for name, ch in channels.items():
        flat = ch.ravel().astype(np.int64).copy()
        m = int(round(rate * flat.size))
        if m:
            idx = rng.permutation(flat.size)[:m]
            bits = rng.integers(0, 2, m)
            flat[idx] = apply_bits(flat[idx], bits, method, rng, max_value)
        out[name] = flat.reshape(ch.shape)
    return out
