"""Загрузка PNG-изображений в унифицированное представление (набор каналов int64)."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

PathOrImage = Union[str, bytes, Image.Image]


class UnsupportedImageError(Exception):
    pass


@dataclass
class ImageData:
    mode: str                       # исходный режим PIL
    width: int
    height: int
    channels: Dict[str, np.ndarray]  # анализируемые каналы (2D, int64)
    max_value: int                  # 255 или 65535
    alpha: Optional[np.ndarray] = None
    palette: Optional[List[int]] = None
    info: Dict = field(default_factory=dict)
    path: Optional[str] = None

    @property
    def n_samples(self) -> int:
        return int(sum(ch.size for ch in self.channels.values()))

    @property
    def is_palette(self) -> bool:
        return self.mode == "P"

    def copy_with(self, channels: Dict[str, np.ndarray]) -> "ImageData":
        return ImageData(self.mode, self.width, self.height, channels, self.max_value,
                         None if self.alpha is None else self.alpha.copy(),
                         self.palette, dict(self.info), self.path)

    def crop_center(self, max_pixels: int) -> "ImageData":
        """Центральный фрагмент не более max_pixels пикселей (для ускорения анализа)."""
        if self.width * self.height <= max_pixels:
            return self
        scale = (max_pixels / (self.width * self.height)) ** 0.5
        return self._crop(max(8, int(self.width * scale)), max(8, int(self.height * scale)))

    def crop_side(self, max_side: int) -> "ImageData":
        return self._crop(min(self.width, max_side), min(self.height, max_side))

    def _crop(self, w: int, h: int) -> "ImageData":
        if w == self.width and h == self.height:
            return self
        x0 = (self.width - w) // 2
        y0 = (self.height - h) // 2
        chans = {k: v[y0:y0 + h, x0:x0 + w].copy() for k, v in self.channels.items()}
        alpha = None if self.alpha is None else self.alpha[y0:y0 + h, x0:x0 + w].copy()
        return ImageData(self.mode, w, h, chans, self.max_value, alpha, self.palette,
                         dict(self.info), self.path)

    def to_pil(self) -> Image.Image:
        if self.mode == "P":
            img = Image.fromarray(self.channels["index"].astype(np.uint8), "P")
            if self.palette:
                img.putpalette(self.palette)
            return img
        if self.mode in ("I;16", "I"):
            return Image.fromarray(np.clip(self.channels["Y"], 0, 65535).astype(np.uint16))
        if self.mode in ("L", "LA"):
            y = self.channels["Y"].astype(np.uint8)
            if self.mode == "LA" and self.alpha is not None:
                return Image.fromarray(np.dstack([y, self.alpha.astype(np.uint8)]), "LA")
            return Image.fromarray(y, "L")
        rgb = np.dstack([self.channels[c].astype(np.uint8) for c in ("R", "G", "B")])
        if self.mode == "RGBA" and self.alpha is not None:
            return Image.fromarray(np.dstack([rgb, self.alpha.astype(np.uint8)]), "RGBA")
        return Image.fromarray(rgb, "RGB")

    def save_png(self, path: str) -> None:
        self.to_pil().save(path, format="PNG")


def _open(src: PathOrImage) -> Image.Image:
    if isinstance(src, Image.Image):
        return src
    if isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    else:
        img = Image.open(src)
    img.load()
    return img


def load_image(src: PathOrImage, include_alpha: bool = False) -> ImageData:
    try:
        img = _open(src)
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedImageError(f"Не удалось декодировать изображение: {exc}") from exc

    path = src if isinstance(src, str) else None
    info = {"format": img.format, "original_mode": img.mode}
    mode = img.mode
    alpha = None
    palette = None
    max_value = 255

    if mode == "1":
        img = img.convert("L")
        mode = "L"
    if mode in ("CMYK", "YCbCr", "LAB", "HSV", "RGBX", "RGBa", "La", "PA"):
        img = img.convert("RGBA" if "A" in mode or "a" in mode else "RGB")
        mode = img.mode

    if mode == "P":
        chans = {"index": np.asarray(img, dtype=np.int64)}
        palette = img.getpalette()
    elif mode == "L":
        chans = {"Y": np.asarray(img, dtype=np.int64)}
    elif mode == "LA":
        arr = np.asarray(img, dtype=np.int64)
        chans = {"Y": arr[..., 0]}
        alpha = arr[..., 1]
    elif mode == "RGB":
        arr = np.asarray(img, dtype=np.int64)
        chans = {"R": arr[..., 0], "G": arr[..., 1], "B": arr[..., 2]}
    elif mode == "RGBA":
        arr = np.asarray(img, dtype=np.int64)
        chans = {"R": arr[..., 0], "G": arr[..., 1], "B": arr[..., 2]}
        alpha = arr[..., 3]
    elif mode.startswith("I;16") or mode == "I":
        arr = np.asarray(img).astype(np.int64)
        arr = np.clip(arr, 0, 65535)
        chans = {"Y": arr}
        mode = "I;16"
        max_value = 65535
    else:
        raise UnsupportedImageError(f"Режим изображения {mode} не поддерживается")

    if include_alpha and alpha is not None and not np.all(alpha == alpha.flat[0]):
        chans["A"] = alpha.copy()

    chans = {k: np.ascontiguousarray(v) for k, v in chans.items()}
    h, w = next(iter(chans.values())).shape
    return ImageData(mode, w, h, chans, max_value, alpha, palette, info, path)
