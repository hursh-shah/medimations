from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


PPM_MAGIC = b"P6"


@dataclass(frozen=True)
class PpmImage:
    width: int
    height: int
    rgb: bytes  # length == width * height * 3


def write_ppm(path: Path, width: int, height: int, rgb: bytes) -> None:
    if len(rgb) != width * height * 3:
        raise ValueError("rgb must be width*height*3 bytes")
    header = b"".join([PPM_MAGIC, b"\n", f"{width} {height}".encode(), b"\n255\n"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + rgb)


def _read_token(data: memoryview, offset: int) -> Tuple[bytes, int]:
    n = len(data)
    while offset < n and data[offset] in b" \t\r\n":
        offset += 1
    if offset >= n:
        raise ValueError("Unexpected EOF while reading PPM header")
    if data[offset] == ord("#"):
        while offset < n and data[offset] != ord("\n"):
            offset += 1
        return _read_token(data, offset)
    start = offset
    while offset < n and data[offset] not in b" \t\r\n":
        offset += 1
    return bytes(data[start:offset]), offset


def read_ppm(path: Path) -> PpmImage:
    raw = path.read_bytes()
    data = memoryview(raw)
    offset = 0
    magic, offset = _read_token(data, offset)
    if magic != PPM_MAGIC:
        raise ValueError(f"Not a binary PPM (P6): {path}")
    w_b, offset = _read_token(data, offset)
    h_b, offset = _read_token(data, offset)
    maxv_b, offset = _read_token(data, offset)
    width = int(w_b)
    height = int(h_b)
    maxv = int(maxv_b)
    if maxv != 255:
        raise ValueError(f"Unsupported maxval {maxv} in {path}")
    expected = width * height * 3
    if len(raw) < expected:
        raise ValueError(f"PPM file too small: got {len(raw)} bytes, expected at least {expected}")

    # Robustly locate pixel data from the end. This avoids ambiguity when the
    # first pixel bytes happen to look like whitespace.
    pixel_start = len(raw) - expected
    if pixel_start < offset:
        raise ValueError(
            f"PPM header parse error: header={offset} bytes but expected pixels start at {pixel_start}"
        )
    pixels = raw[pixel_start:]
    return PpmImage(width=width, height=height, rgb=pixels)


def mean_rgb(ppm: PpmImage, stride: int = 16) -> Tuple[float, float, float]:
    if stride < 1:
        stride = 1
    total_r = 0
    total_g = 0
    total_b = 0
    count = 0
    rgb = ppm.rgb
    for i in range(0, len(rgb), 3 * stride):
        total_r += rgb[i]
        total_g += rgb[i + 1]
        total_b += rgb[i + 2]
        count += 1
    if count == 0:
        return 0.0, 0.0, 0.0
    return total_r / count, total_g / count, total_b / count


def find_reddest_pixel(ppm: PpmImage, stride: int = 4) -> Tuple[int, int, float]:
    """
    Returns (x, y, redness_score) of the "reddest" sampled pixel.
    Redness score is a simple (r - (g+b)/2) proxy in [0, 255].
    """
    if stride < 1:
        stride = 1
    best_score = float("-inf")
    best_x = 0
    best_y = 0
    w = ppm.width
    h = ppm.height
    rgb = ppm.rgb
    for y in range(0, h, stride):
        row = y * w * 3
        for x in range(0, w, stride):
            i = row + x * 3
            r = rgb[i]
            g = rgb[i + 1]
            b = rgb[i + 2]
            score = float(r) - (float(g) + float(b)) / 2.0
            if score > best_score:
                best_score = score
                best_x = x
                best_y = y
    return best_x, best_y, best_score


def red_centroid(ppm: PpmImage, stride: int = 2, threshold: float = 50.0) -> Tuple[float, float, float]:
    """
    Returns (x, y, mean_redness) centroid of pixels whose redness exceeds threshold.
    If no pixels qualify, returns (-1, -1, -inf).
    """
    if stride < 1:
        stride = 1
    w = ppm.width
    h = ppm.height
    rgb = ppm.rgb

    sum_w = 0.0
    sum_x = 0.0
    sum_y = 0.0
    sum_redness = 0.0
    count = 0

    for y in range(0, h, stride):
        row = y * w * 3
        for x in range(0, w, stride):
            i = row + x * 3
            r = rgb[i]
            g = rgb[i + 1]
            b = rgb[i + 2]
            redness = float(r) - (float(g) + float(b)) / 2.0
            if redness <= threshold:
                continue
            wgt = redness
            sum_w += wgt
            sum_x += wgt * x
            sum_y += wgt * y
            sum_redness += redness
            count += 1

    if count == 0 or sum_w <= 0.0:
        return -1.0, -1.0, float("-inf")

    return (sum_x / sum_w), (sum_y / sum_w), (sum_redness / count)
