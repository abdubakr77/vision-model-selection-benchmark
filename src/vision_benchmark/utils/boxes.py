"""Box helpers: YOLO boxes (normalized cx, cy, w, h) to pixels, and one row per box."""

import json

import pandas as pd


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
    """One row per box from a split.csv style DataFrame (columns image_id, split, label, width, height, boxes).

    Returns columns: image_id, split, label, box_idx, x0, y0, x1, y1, w_px, h_px, area_px.
    Images without boxes do not appear.
    """
    rows = []
    for _, r in data.iterrows():
        if not isinstance(r["boxes"], str):
            continue
        boxes = normalized_to_pixels(json.loads(r["boxes"]), int(r["width"]), int(r["height"]))
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            rows.append(dict(image_id=r["image_id"], split=r["split"], label=r["label"], box_idx=k,
                             x0=x0, y0=y0, x1=x1, y1=y1, w_px=x1 - x0, h_px=y1 - y0,
                             area_px=(x1 - x0) * (y1 - y0)))
    return pd.DataFrame(rows)
