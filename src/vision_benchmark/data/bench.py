"""Load raw VisA samples upright and resized, and write them into the bench folder (color + gray, YOLO layout).

bench/
  split.csv, meta.json
  color/data.yaml, color/images/<split>/, color/labels/<split>/
  gray/data.yaml,  gray/images/<split>/,  gray/labels/<split>/
  masks/<split>/   (defect images only, 0 or 255)
"""

import os

import cv2

from ..utils.boxes import pixels_to_normalized
from .visa import read_mask


def load_upright(visa_dir, path, mask_path, rotate, size):
    """Raw image (and mask if mask_path is set) turned upright when rotate is True, then resized to size (w, h)."""
    img = cv2.imread(os.path.join(visa_dir, path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    mask = read_mask(os.path.join(visa_dir, mask_path)) if mask_path else None
    if rotate:  # the mask turns with the image, so defects stay in place
        img = cv2.rotate(img, cv2.ROTATE_180)
        mask = None if mask is None else cv2.rotate(mask, cv2.ROTATE_180)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)  # nearest keeps the mask 0/1
    return img, mask


def save_sample(bench_dir, split, image_id, img_bgr, mask=None, boxes=()):
    """Write one sample: color and gray JPEG, the same YOLO label file next to each, and the mask (defects only).

    Returns the written paths relative to bench_dir, with "/" separators so they work on Windows and Linux.
    """
    h, w = img_bgr.shape[:2]
    label_text = "\n".join("0 " + " ".join(f"{v:.6f}" for v in b) for b in pixels_to_normalized(boxes, w, h))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for mode, img in [("color", img_bgr), ("gray", gray)]:
        os.makedirs(os.path.join(bench_dir, mode, "images", split), exist_ok=True)
        os.makedirs(os.path.join(bench_dir, mode, "labels", split), exist_ok=True)
        cv2.imwrite(os.path.join(bench_dir, mode, "images", split, image_id + ".jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(bench_dir, mode, "labels", split, image_id + ".txt"), "w") as f:
            f.write(label_text)  # empty file = normal image, YOLO treats it as background

    mask_rel = ""
    if mask is not None:
        os.makedirs(os.path.join(bench_dir, "masks", split), exist_ok=True)
        mask_rel = f"masks/{split}/{image_id}.png"
        cv2.imwrite(os.path.join(bench_dir, mask_rel), mask * 255)
    return dict(path=f"color/images/{split}/{image_id}.jpg", path_gray=f"gray/images/{split}/{image_id}.jpg",
                mask_path=mask_rel)


def write_yolo_yaml(bench_dir):
    """One data.yaml per image mode. No "path" key: YOLO then uses the folder of the yaml, so it works anywhere."""
    for mode in ["color", "gray"]:
        with open(os.path.join(bench_dir, mode, "data.yaml"), "w") as f:
            f.write("train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n")
