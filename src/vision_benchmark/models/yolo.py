"""YOLO detector used as an image-level defect model."""

import json
import os

import pandas as pd

from ..utils.boxes import normalized_to_pixels


def predict_yolo(model, data, bench_dir, mode, imgsz=1024, batch=16, min_conf=0.01):
    """Run a trained Ultralytics model on every image in data and keep all boxes above min_conf.

    data: rows of bench/split.csv (for example one split). mode "gray" reads path_gray, "color" reads path.
    Returns one row per image with the true boxes and the predicted boxes, both in pixels
    (x0, y0, x1, y1; predicted boxes also carry their confidence). Any confidence threshold can be
    applied later with apply_conf, without running the model again.
    score = highest box confidence in the image (0 if no box), used as the image-level defect score.
    """
    column = "path_gray" if mode == "gray" else "path"
    records = data.to_dict("records")
    rows = []
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        paths = [os.path.join(bench_dir, r[column]) for r in chunk]
        results = model.predict(paths, imgsz=imgsz, conf=min_conf, verbose=False)
        for r, path, res in zip(chunk, paths, results):
            h, w = res.orig_shape
            true_boxes = [list(map(float, b)) for b in normalized_to_pixels(json.loads(r["boxes"]), w, h)]
            pred_boxes = [box + [conf] for box, conf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())]
            rows.append({"image_id": r["image_id"], "label": r["label"], "defect_type": r["defect_type"],
                         "has_small_box": r["has_small_box"], "path": path, "true_boxes": true_boxes,
                         "pred_boxes": pred_boxes, "score": max((b[4] for b in pred_boxes), default=0.0)})
    return pd.DataFrame(rows)
