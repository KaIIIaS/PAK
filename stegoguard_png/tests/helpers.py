import numpy as np

from stegoguard.image_io import load_image
from stegoguard.samples import cover_camera, cover_hmi, cover_thermal, png_bytes


def camera_png(seed=1, h=256, w=320):
    return png_bytes(cover_camera(np.random.default_rng(seed), h, w))


def hmi_png(seed=1, h=256, w=320):
    return png_bytes(cover_hmi(np.random.default_rng(seed), h, w))


def thermal_png(seed=1, h=256, w=320):
    return png_bytes(cover_thermal(np.random.default_rng(seed), h, w))


def image(data):
    return load_image(data)
