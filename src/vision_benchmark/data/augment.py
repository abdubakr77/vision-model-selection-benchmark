"""Train-only augmentation for upright board images: random safe zoom in, brightness and contrast.

The zoom never cuts the board, the pins under it or any defect box, so a defective image keeps its defect.
"""

import cv2
import numpy as np

from .orientation import find_board

PIN_FRAC = 0.45     # pins stick out about 0.3 x board height below the board, 0.45 keeps a margin
BOARD_MARGIN = 0.03  # extra margin around the board, as a fraction of the board width


def safe_region(img_bgr, boxes=()):
    """Box (x0, y0, x1, y1) that every zoom crop must contain: board + pins below it + all defect boxes.

    The image must be upright (pins at the bottom).
    """
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = find_board(img_bgr)
    m = int(BOARD_MARGIN * (x1 - x0))
    x0, y0, x1 = x0 - m, y0 - m, x1 + m
    y1 = y1 + int(PIN_FRAC * (y1 - y0))
    for bx0, by0, bx1, by1 in boxes:
        x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def max_zoom(img_shape, safe):
    """Largest zoom in factor whose crop (same aspect ratio as the image) still contains the safe region."""
    h, w = img_shape[:2]
    sx0, sy0, sx1, sy1 = safe
    return max(1.0, min(w / (sx1 - sx0), h / (sy1 - sy0)))


def random_zoom(img, mask, safe, rng):
    """Zoom in by a random factor between 1 (no zoom) and max_zoom, at a random position that keeps safe inside.

    The crop is resized back to the original size. mask (or None) gets the same crop.
    """
    h, w = img.shape[:2]
    z = rng.uniform(1.0, max_zoom(img.shape, safe))
    sx0, sy0, sx1, sy1 = safe
    cw = max(int(w / z), sx1 - sx0)  # rounding must never make the crop smaller than the safe region
    ch = max(int(h / z), sy1 - sy0)
    # left edge must be <= safe left and right edge >= safe right; same for top and bottom
    cx0 = int(rng.integers(max(0, sx1 - cw), min(sx0, w - cw) + 1))
    cy0 = int(rng.integers(max(0, sy1 - ch), min(sy0, h - ch) + 1))
    img = cv2.resize(img[cy0:cy0 + ch, cx0:cx0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask = cv2.resize(mask[cy0:cy0 + ch, cx0:cx0 + cw], (w, h), interpolation=cv2.INTER_NEAREST)
    return img, mask, z


def random_brightness_contrast(img, rng, contrast=(0.8, 1.2), brightness=(-25, 25)):
    """new pixel = old pixel * contrast + brightness, clipped to 0..255."""
    a = rng.uniform(*contrast)
    b = rng.uniform(*brightness)
    return cv2.convertScaleAbs(img, alpha=a, beta=b), a, b


def augment(img, mask, boxes, rng, p_zoom=0.7):
    """One augmented copy: zoom in with probability p_zoom (random strength), then brightness and contrast.

    Returns the new image, the new mask (or None) and a dict of what was applied.
    """
    zoom = 1.0
    if rng.random() < p_zoom:
        img, mask, zoom = random_zoom(img, mask, safe_region(img, boxes), rng)
    img, a, b = random_brightness_contrast(img, rng)
    return img, mask, dict(zoom=round(zoom, 3), contrast=round(a, 3), brightness=round(b, 1))
