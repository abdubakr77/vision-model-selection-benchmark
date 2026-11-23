"""Plotting helpers for EDA and failure analysis."""

import json
import os

import cv2
import matplotlib.pyplot as plt
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
