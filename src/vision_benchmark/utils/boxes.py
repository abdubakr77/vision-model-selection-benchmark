"""Box helpers: masks to boxes, pixel <-> YOLO (normalized cx, cy, w, h), and one row per box."""

import json

import cv2
import numpy as np
import pandas as pd


def mask_to_boxes(mask, dilate_px=9, min_area=15):
    """Binary mask (defect > 0) to pixel boxes (x0, y0, x1, y1).

    Nearby fragments are joined by a small dilation, specks under min_area pixels are dropped,
    and boxes that overlap or sit inside each other are merged into one.
    """
    m = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(m, k), connectivity=8)
    boxes = [[x, y, x + w, y + h] for x, y, w, h, area in stats[1:].tolist() if area >= min_area]

    # merge overlapping boxes until no two boxes overlap
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(b) for b in boxes]


def pixels_to_normalized(boxes, width, height):
    """Pixel boxes (x0, y0, x1, y1) to YOLO boxes (normalized cx, cy, w, h)."""
    return [[(x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height]
            for x0, y0, x1, y1 in boxes]


def normalized_to_pixels(boxes, width, height):
    """YOLO boxes (normalized cx, cy, w, h) to pixel corners (x0, y0, x1, y1), clipped to the image."""
    out = []
    for cx, cy, bw, bh in boxes:
        x0 = max(0, int(round((cx - bw / 2) * width)))
        y0 = max(0, int(round((cy - bh / 2) * height)))
        x1 = min(width, int(round((cx + bw / 2) * width)))
        y1 = min(height, int(round((cy + bh / 2) * height)))
        out.append((x0, y0, x1, y1))
    return out


def boxes_table(data: pd.DataFrame):
    """One row per box from a DataFrame with columns image_id, label, width, height, boxes (YOLO, JSON string).

    Returns columns: image_id, split (empty if missing), label, box_idx, x0, y0, x1, y1, w_px, h_px, area_px.
    Images without boxes do not appear.
    """
    rows = []
    for _, r in data.iterrows():
        if not isinstance(r["boxes"], str):
            continue
        boxes = normalized_to_pixels(json.loads(r["boxes"]), int(r["width"]), int(r["height"]))
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            rows.append(dict(image_id=r["image_id"], split=r.get("split", ""), label=r["label"], box_idx=k,
                             x0=x0, y0=y0, x1=x1, y1=y1, w_px=x1 - x0, h_px=y1 - y0,
                             area_px=(x1 - x0) * (y1 - y0)))
    return pd.DataFrame(rows)
