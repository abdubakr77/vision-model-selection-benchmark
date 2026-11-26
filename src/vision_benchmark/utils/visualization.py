"""Plotting helpers for EDA and failure analysis."""

import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .boxes import normalized_to_pixels

CROP_PAD_PX = 20  # extra pixels kept around each box when cropping, tiny defects need context


def _read_rgb_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_sample_images(data: pd.DataFrame, num_images=3, filter_data=None, show_boxes=True,
                       crop_boxes=False, save_plot=None, base_dir=None, random_state=42):
    """Show random images from a split.csv style DataFrame with their image_id and label.

    data: DataFrame with columns image_id, label, path and boxes (YOLO format, JSON string).
    num_images: how many random images to show.
    filter_data: [column, value] to keep only matching rows, for example ["label", 1].
    show_boxes: draw the defect boxes on the full image.
    crop_boxes: also show every box as a crop (pixel coordinates) next to its image.
    save_plot: None does not save, True saves sample_images.png in os.getcwd(),
        a string saves to that file path (folders are created).
    base_dir: folder the "path" column is relative to (the bench folder).
    random_state: seed for the random sample, so the same images come back every run.
    """
    
    df = data[data["path"].notna()] # Skip NaN Values
    if filter_data is not None:
        column, value = filter_data
        df = df[df[column] == value]
    if df.empty:
        raise ValueError("no rows left to show after filtering")
    df = df.sample(n=min(num_images, len(df)), random_state=random_state)

    rows = []
    for _, r in df.iterrows():
        rel = str(r["path"]).replace("\\", os.sep).replace("/", os.sep)  # csv may hold Windows paths
        img = _read_rgb_image(os.path.join(base_dir, rel) if base_dir else rel)
        h, w = img.shape[:2]
        boxes = normalized_to_pixels(json.loads(r["boxes"]), w, h) if isinstance(r["boxes"], str) else []
        rows.append((r, img, boxes))

    n_cols = 1 + (max(len(b) for _, _, b in rows) if crop_boxes else 0)
    n_cols = max(n_cols, 1)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(5 * n_cols, 4 * len(rows)), squeeze=False)

    for i, (r, img, boxes) in enumerate(rows):
        shown = img.copy()
        # For Showing the boxes
        if show_boxes:
            thickness = max(2, img.shape[1] // 400)
            for x0, y0, x1, y1 in boxes:
                cv2.rectangle(shown, (x0, y0), (x1, y1), (255, 0, 0), thickness)
        axes[i, 0].imshow(shown)
        axes[i, 0].set_title(f"{r['image_id']} | label {r['label']}", fontsize=10)

        # Crop For Each Box 
        for j in range(1, n_cols):
            ax = axes[i, j]
            if crop_boxes and j - 1 < len(boxes):
                x0, y0, x1, y1 = boxes[j - 1]
                ih, iw = img.shape[:2]
                crop = img[max(0, y0 - CROP_PAD_PX):min(ih, y1 + CROP_PAD_PX),
                           max(0, x0 - CROP_PAD_PX):min(iw, x1 + CROP_PAD_PX)]
                ax.imshow(crop)
                ax.set_title(f"box {j}: {x1 - x0}x{y1 - y0} px", fontsize=10)
            ax.axis("off")
        axes[i, 0].axis("off")

    fig.tight_layout()

    if save_plot is not None and save_plot is not False:
        out = os.path.join(os.getcwd(), "sample_images.png") if save_plot is True else str(save_plot)
        folder = os.path.dirname(os.path.abspath(out))
        os.makedirs(folder, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"saved: {out}")

    plt.show()


def show_thumbnail_grid(paths, n_cols=20, thumb_width=160, title=None, save_plot=None):
    """Many images as small thumbnails in one grid, to scan a whole group by eye (for example all rotated boards).

    paths: full image paths. save_plot follows the show_sample_images rules.
    """
    thumbs = []
    for p in paths:
        img = _read_rgb_image(p)
        h, w = img.shape[:2]
        thumbs.append(cv2.resize(img, (thumb_width, int(h * thumb_width / w)), interpolation=cv2.INTER_AREA))
    th, tw = thumbs[0].shape[:2]
    n_rows = (len(thumbs) + n_cols - 1) // n_cols
    sheet = np.full((n_rows * th, n_cols * tw, 3), 255, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, n_cols)
        sheet[r * th:r * th + t.shape[0], c * tw:c * tw + t.shape[1]] = t[:th, :tw]

    fig = plt.figure(figsize=(n_cols * tw / 100, n_rows * th / 100 + 0.6))
    plt.imshow(sheet)
    plt.axis("off")
    if title:
        plt.title(title)
    if save_plot is not None and save_plot is not False:
        out = os.path.join(os.getcwd(), "thumbnails.png") if save_plot is True else str(save_plot)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=100, bbox_inches="tight")
        print(f"saved: {out}")
    plt.show()


def show_color_vs_gray(data: pd.DataFrame, num_images=4, filter_data=None, save_plot=None, base_dir=None,
                       random_state=42):
    """For random images, show the full image and the crop of its largest box, in color and in grayscale.

    Same data, filter_data, save_plot, base_dir and random_state rules as show_sample_images.
    Images without boxes show the full image only.
    """
    df = data[data["path"].notna()]
    if filter_data is not None:
        column, value = filter_data
        df = df[df[column] == value]
    if df.empty:
        raise ValueError("no rows left to show after filtering")
    df = df.sample(n=min(num_images, len(df)), random_state=random_state)

    fig, axes = plt.subplots(len(df), 4, figsize=(18, 3.6 * len(df)), squeeze=False)
    for i, (_, r) in enumerate(df.iterrows()):
        rel = str(r["path"]).replace("\\", os.sep).replace("/", os.sep)
        img = _read_rgb_image(os.path.join(base_dir, rel) if base_dir else rel)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = img.shape[:2]
        boxes = normalized_to_pixels(json.loads(r["boxes"]), w, h) if isinstance(r["boxes"], str) else []

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{r['image_id']} | color", fontsize=10)
        axes[i, 1].imshow(gray, cmap="gray")
        axes[i, 1].set_title("grayscale", fontsize=10)
        if boxes:
            x0, y0, x1, y1 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            ys = slice(max(0, y0 - CROP_PAD_PX), min(h, y1 + CROP_PAD_PX))
            xs = slice(max(0, x0 - CROP_PAD_PX), min(w, x1 + CROP_PAD_PX))
            axes[i, 2].imshow(img[ys, xs])
            axes[i, 2].set_title(f"largest box {x1 - x0}x{y1 - y0} px | color", fontsize=10)
            axes[i, 3].imshow(gray[ys, xs], cmap="gray")
            axes[i, 3].set_title("grayscale", fontsize=10)
        for ax in axes[i]:
            ax.axis("off")

    fig.tight_layout()
    if save_plot is not None and save_plot is not False:
        out = os.path.join(os.getcwd(), "color_vs_gray.png") if save_plot is True else str(save_plot)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"saved: {out}")
    plt.show()


def _save_figure(fig, save_plot, default_name):
    """Save rule shared by the training plots: None/False no save, True to os.getcwd(), str to that path."""
    if save_plot is None or save_plot is False:
        return
    out = os.path.join(os.getcwd(), default_name) if save_plot is True else str(save_plot)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved: {out}")


def show_detections(df, num_images=4, outcome=None, random_state=42, save_plot=None):
    """True boxes (green) next to predicted boxes (red, with confidence) for images from apply_conf.

    outcome: "TP", "FN", "FP" or "TN" to show only that kind, None for any.
    Each row is the full image and a zoom on the area of the true and predicted boxes,
    so the difference between them is visible. The title gives the image score and each
    true box's best IoU with a prediction (0 = missed).
    """
    rows = df if outcome is None else df[df["outcome"] == outcome]
    if rows.empty:
        print(f"no images with outcome {outcome}")
        return
    rows = rows.sample(n=min(num_images, len(rows)), random_state=random_state)

    fig, axes = plt.subplots(len(rows), 2, figsize=(16, 5.5 * len(rows)), squeeze=False)
    for ax_row, (_, r) in zip(axes, rows.iterrows()):
        img = _read_rgb_image(r["path"])
        thick = max(2, img.shape[1] // 400)
        for x0, y0, x1, y1 in r["true_boxes"]:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 0), thick)
        for x0, y0, x1, y1, c in r["kept_boxes"]:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (230, 0, 0), thick)
            cv2.putText(img, f"{c:.2f}", (int(x0), max(int(y0) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 0, 0), 2)

        ious = ", ".join(f"{i:.2f}" for i in r["best_iou"]) or "-"
        ax_row[0].imshow(img)
        ax_row[0].set_title(f"{r['image_id']} | {r['outcome']} | {r['defect_type']} | score {r['score']:.2f}\n"
                            f"true {r['n_true']} (green), predicted {r['n_pred']} (red) | best IoU per true box: {ious}")

        # zoom on every box of the image (true and predicted), with some margin
        boxes = [b[:4] for b in r["true_boxes"]] + [b[:4] for b in r["kept_boxes"]]
        if boxes:
            pad = 60
            x0 = max(int(min(b[0] for b in boxes)) - pad, 0)
            y0 = max(int(min(b[1] for b in boxes)) - pad, 0)
            x1 = min(int(max(b[2] for b in boxes)) + pad, img.shape[1])
            y1 = min(int(max(b[3] for b in boxes)) + pad, img.shape[0])
            ax_row[1].imshow(img[y0:y1, x0:x1])
            ax_row[1].set_title("zoom on the boxes")
        for ax in ax_row:
            ax.axis("off")
    fig.tight_layout()
    _save_figure(fig, save_plot, "detections.png")
    plt.show()


def show_crop_examples(raw_images, crop_boxes, cropped_images, titles, save_plot=None):
    """Each row: the upright raw image with its crop box (red) and the cropped, resized result.

    raw_images and cropped_images are BGR arrays; crop_boxes are (x0, y0, x1, y1) in raw pixels.
    """
    fig, axes = plt.subplots(len(raw_images), 2, figsize=(16, 5 * len(raw_images)), squeeze=False)
    for ax_row, raw, box, cropped, title in zip(axes, raw_images, crop_boxes, cropped_images, titles):
        shown = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        cv2.rectangle(shown, box[:2], box[2:], (230, 0, 0), max(2, raw.shape[1] // 300))
        ax_row[0].imshow(shown)
        ax_row[0].set_title(f"{title} | raw {raw.shape[1]} x {raw.shape[0]}, crop box in red")
        ax_row[1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        ax_row[1].set_title(f"cropped and resized to {cropped.shape[1]} x {cropped.shape[0]}")
        for ax in ax_row:
            ax.axis("off")
    fig.tight_layout()
    _save_figure(fig, save_plot, "crop_examples.png")
    plt.show()
