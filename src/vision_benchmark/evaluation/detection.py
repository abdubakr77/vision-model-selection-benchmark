"""Detector results at a chosen confidence: image outcome, box overlap with the truth, and a short report."""

import numpy as np
import pandas as pd


def box_iou(a, b):
    """Intersection over union of two pixel boxes (x0, y0, x1, y1)."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def apply_conf(pred_df, conf, iou_thr=0.5):
    """Keep predicted boxes with confidence >= conf and score every image.

    pred_df: output of predict_yolo. Adds:
      kept_boxes  predicted boxes that pass conf
      pred        1 if any box passes conf (image called defective)
      outcome     TP found defect, FN missed defect, FP false alarm on a normal image, TN correct normal
      best_iou    for every true box, its best IoU with a kept box (0 = no box touches it)
      boxes_hit   true boxes with best IoU >= iou_thr
    """
    df = pred_df.copy()
    df["kept_boxes"] = [[b for b in boxes if b[4] >= conf] for boxes in df["pred_boxes"]]
    df["n_true"] = df["true_boxes"].apply(len)
    df["n_pred"] = df["kept_boxes"].apply(len)
    df["pred"] = (df["n_pred"] > 0).astype(int)
    df["outcome"] = np.select([(df["label"] == 1) & (df["pred"] == 1), (df["label"] == 1) & (df["pred"] == 0),
                               (df["label"] == 0) & (df["pred"] == 1)], ["TP", "FN", "FP"], "TN")
    df["best_iou"] = [[max((box_iou(t, p) for p in kept), default=0.0) for t in true]
                      for true, kept in zip(df["true_boxes"], df["kept_boxes"])]
    df["boxes_hit"] = df["best_iou"].apply(lambda ious: sum(i >= iou_thr for i in ious))
    df["conf"] = conf
    return df


def detection_summary(df, low_conf=0.5):
    """Counts a person can read at a glance, from the output of apply_conf."""
    defect, normal = df[df["label"] == 1], df[df["label"] == 0]
    found = defect[defect["outcome"] == "TP"]
    hit_ious = [i for ious in defect["best_iou"] for i in ious if i > 0]
    rows = {
        "confidence threshold": round(float(df["conf"].iloc[0]), 3),
        "defective images": len(defect),
        "  found (TP)": len(found),
        "  missed (FN)": int((defect["outcome"] == "FN").sum()),
        "  found with low confidence (< %.2f)" % low_conf: int((found["score"] < low_conf).sum()),
        "  found, but no box on a true defect": int((found["boxes_hit"] == 0).sum()),
        "normal images": len(normal),
        "  correct (TN)": int((normal["outcome"] == "TN").sum()),
        "  false alarms (FP)": int((normal["outcome"] == "FP").sum()),
        "true defect boxes": int(defect["n_true"].sum()),
        "  boxes hit (IoU >= threshold)": int(defect["boxes_hit"].sum()),
        "  mean IoU of touched boxes": round(float(np.mean(hit_ious)), 3) if hit_ious else 0.0,
        "image recall": round(len(found) / max(len(defect), 1), 3),
        "image false alarm rate": round(float((normal["outcome"] == "FP").mean()), 3) if len(normal) else 0.0,
    }
    return pd.Series(rows, name="value", dtype=object).to_frame()  # object keeps counts as whole numbers


def recall_by_type(df):
    """Found / missed defective images per defect type and per small-box flag."""
    defect = df[df["label"] == 1]
    parts = []
    for column in ["defect_type", "has_small_box"]:
        t = defect.groupby(column)["outcome"].value_counts().unstack(fill_value=0)
        t = t.reindex(columns=["TP", "FN"], fill_value=0)
        t["recall"] = (t["TP"] / (t["TP"] + t["FN"])).round(3)
        t.index = [f"{column} = {v}" for v in t.index]
        parts.append(t)
    return pd.concat(parts)
