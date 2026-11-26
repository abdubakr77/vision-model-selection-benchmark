"""Fixed crop around the board: the same rule for every image (train, val, test and any new image).

Measured on the 650 used pcb1 images (raw 1404 x 1070, upright): the board is about 947 x 421 px,
defects never reach more than 0.014 board heights above it or past its sides, and the lowest defect
(a bent or missing pin) ends 0.40 board heights under it.
"""

import os

import cv2
import numpy as np

from .orientation import find_board
from .visa import read_mask

CROP_MARGIN = 0.02    # margin left, right and above the board, as a fraction of the board width
CROP_BELOW = 0.45     # room under the board for the pins, as a fraction of the board height (pins end at about 0.40)


def board_crop_box(img_bgr, aspect):
    """Crop box (x0, y0, x1, y1) around the board of an upright image.

    Board + CROP_MARGIN on the left, right and top + CROP_BELOW under it, then widened (or made taller)
    evenly on both sides until width / height == aspect, so every crop resizes to the same size without stretching.
    """
    h, w = img_bgr.shape[:2]
    bx0, by0, bx1, by1 = find_board(img_bgr)
    m = CROP_MARGIN * (bx1 - bx0)
    x0, y0, x1, y1 = bx0 - m, by0 - m, bx1 + m, by1 + CROP_BELOW * (by1 - by0)

    cw, ch = x1 - x0, y1 - y0
    if cw / ch < aspect:      # too narrow: add width on both sides
        extra = ch * aspect - cw
        x0, x1 = x0 - extra / 2, x1 + extra / 2
    else:                     # too wide: add height on both sides
        extra = cw / aspect - ch
        y0, y1 = y0 - extra / 2, y1 + extra / 2

    # slide the box back inside the image if it sticks out (the size stays the same)
    dx = max(0.0, -x0) - max(0.0, x1 - w)
    dy = max(0.0, -y0) - max(0.0, y1 - h)
    return int(round(x0 + dx)), int(round(y0 + dy)), int(round(x1 + dx)), int(round(y1 + dy))


def load_cropped(visa_dir, path, mask_path, rotate, size):
    """Raw image (and mask if mask_path is set): turned upright when rotate is True, cropped to the board
    at full resolution, then resized to size (w, h).

    Returns (image, mask or None, crop box in raw upright pixels, defect pixels left outside the crop).
    """
    img = cv2.imread(os.path.join(visa_dir, path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    mask = read_mask(os.path.join(visa_dir, mask_path)) if mask_path else None
    if rotate:  # the mask turns with the image, so defects stay in place
        img = cv2.rotate(img, cv2.ROTATE_180)
        mask = None if mask is None else cv2.rotate(mask, cv2.ROTATE_180)

    x0, y0, x1, y1 = board_crop_box(img, size[0] / size[1])
    img = cv2.resize(img[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
    lost = 0
    if mask is not None:
        lost = int(mask.sum() - mask[y0:y1, x0:x1].sum())  # must be 0: no defect may be cut off
        mask = cv2.resize(mask[y0:y1, x0:x1], size, interpolation=cv2.INTER_NEAREST)  # nearest keeps 0/1
    return img, mask, (x0, y0, x1, y1), lost
