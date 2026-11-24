"""Read the raw VisA annotation csv and masks for one category."""

import os

import cv2
import numpy as np
import pandas as pd


def load_visa_annotations(visa_dir, category="pcb1"):
    """Raw annotation table of one VisA category, read from <visa_dir>/<category>/image_anno.csv.

    Returns columns: image_id, label (0 normal, 1 defect), defect_type ("normal", "bent", "melt,scratch", ...),
    path and mask_path (relative to visa_dir, "/" separators, mask_path empty for normal images).
    """
    df = pd.read_csv(os.path.join(visa_dir, category, "image_anno.csv"))
    df = df.rename(columns={"image": "path", "mask": "mask_path", "label": "defect_type"})
    df["mask_path"] = df["mask_path"].fillna("")
    df["label"] = (df["defect_type"] != "normal").astype(int)
    stem = df["path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
    df["image_id"] = category + "_" + df["label"].map({0: "normal", 1: "defect"}) + "_" + stem
    return df[["image_id", "label", "defect_type", "path", "mask_path"]]


def read_mask(path):
    """VisA mask as uint8 0/1. Pixel values are defect type ids (1, 3, 4 ...), not 255, so use > 0."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    return (m > 0).astype(np.uint8)
