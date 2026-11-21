# 01_prepare_data.py
# Vision Model Selection Benchmark | Phase 1, Step A
# Downloads VisA (one category), builds ONE fixed split used by every model,
# derives YOLO boxes from the masks, saves per-image metadata.
# CPU only. Do not turn on the GPU for this notebook.
#
# Usage (Kaggle cell or terminal):
#   python 01_prepare_data.py [input_dir] [output_dir]
# Optional env var: VISA_CATEGORY (default pcb1)
# Note: VisA masks hold class ids (0, 1, 3, 4 ...), they look black but are not empty.

import os
import sys
import json
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


SEED = 42
seed_everything(seed=SEED)


def _arg(i, default):
    # Safe in notebooks: ignores "-f kernel.json" style arguments
    if len(sys.argv) > i:
        a = sys.argv[i]
        if a and not a.startswith("-") and not a.endswith(".json"):
            return a
    return default


ON_KAGGLE = os.path.isdir("/kaggle/working")
IN_DIR = Path(_arg(1, "/kaggle/input" if ON_KAGGLE else "./data_in"))
OUT_DIR = Path(_arg(2, "/kaggle/working/bench" if ON_KAGGLE else "./bench"))

# ---------------- config ----------------
CATEGORY = os.environ.get("VISA_CATEGORY", "pcb1")
VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
MAX_SIDE = 1024          # every image is stored with its longest side <= 1024
# split sizes (seed 42). Remaining defects go to train, remaining normals are marked "unused".
# test is close to a realistic ratio (1:10) so false alarms can be measured,
# val has a similar ratio so thresholds tuned on val transfer to test (never tune on test),
# train is kept near 1:3 so the supervised models see enough defects.
N_TEST_DEFECT, N_TEST_NORMAL = 30, 300
N_VAL_DEFECT, N_VAL_NORMAL = 20, 100
N_TRAIN_NORMAL = 150
BOX_DILATE_PX = 9        # merge nearby mask fragments into one box
BOX_MIN_AREA = 15        # ignore specks smaller than this (pixels, after resize)
KEEP_TAR = False
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

RAW_DIR = OUT_DIR / "raw"
YOLO_DIR = OUT_DIR / "yolo"
MASK_DIR = OUT_DIR / "masks"
for d in [OUT_DIR, RAW_DIR, YOLO_DIR, MASK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[cfg] category={CATEGORY} in={IN_DIR} out={OUT_DIR}")


# ---------------- locate or download the data ----------------
def find_category_dir(root, category, max_depth=5):
    root = Path(root)
    if not root.exists():
        return None
    base = len(root.parts)
    for dirpath, dirnames, _ in os.walk(root):
        p = Path(dirpath)
        if len(p.parts) - base > max_depth:
            dirnames[:] = []
            continue
        if p.name.lower() == category.lower() and (p / "Data" / "Images").is_dir():
            return p
    return None


cat_dir = find_category_dir(IN_DIR, CATEGORY) or find_category_dir(RAW_DIR, CATEGORY)

if cat_dir is None:
    tar_path = RAW_DIR / "VisA_20220922.tar"
    if not tar_path.exists():
        print("[data] downloading VisA tar (about 2 GB), this takes a few minutes ...")
        part = tar_path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(VISA_URL, part)
            part.rename(tar_path)
        except Exception as e:
            print(f"[error] download failed: {e}")
            print("Turn on Internet in the Kaggle notebook settings (phone verification needed),")
            print("or add VisA as a Kaggle dataset and pass its folder as the first argument.")
            sys.exit(1)
    print(f"[data] extracting only '{CATEGORY}' from the tar ...")
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf if f"/{CATEGORY}/" in "/" + m.name]
        if not members:
            print(f"[error] no members found for category '{CATEGORY}'")
            sys.exit(1)
        try:
            tf.extractall(RAW_DIR, members=members, filter="data")
        except TypeError:
            tf.extractall(RAW_DIR, members=members)
    if not KEEP_TAR:
        try:
            tar_path.unlink()
        except Exception:
            pass
    cat_dir = find_category_dir(RAW_DIR, CATEGORY)

if cat_dir is None:
    print("[error] could not find <category>/Data/Images after extraction")
    sys.exit(1)
print(f"[data] using {cat_dir}")

img_root = cat_dir / "Data" / "Images"
mask_root = cat_dir / "Data" / "Masks"


def list_images(folder):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


normal_files = list_images(img_root / "Normal")
defect_files = list_images(img_root / "Anomaly")
print(f"[data] found {len(normal_files)} normal, {len(defect_files)} defect images")
if len(defect_files) < 20 or len(normal_files) < 40:
    print("[error] too few images, check the folder layout")
    sys.exit(1)


def find_mask(img_path):
    base = mask_root / "Anomaly"
    for ext in [".png", ".PNG", ".jpg", ".JPG", ".bmp"]:
        cand = base / (img_path.stem + ext)
        if cand.exists():
            return cand
    return None


# keep only defect images that have a mask (needed for boxes)
defect_ok = []
for p in defect_files:
    if find_mask(p) is not None:
        defect_ok.append(p)
    else:
        print(f"[warn] no mask for {p.name}, skipped")
defect_files = defect_ok

# ---------------- fixed split (seed 42) ----------------
rng = random.Random(SEED)
rng.shuffle(normal_files)
rng.shuffle(defect_files)

if len(defect_files) <= N_TEST_DEFECT + N_VAL_DEFECT:
    print("[error] not enough defect images for the configured split")
    sys.exit(1)
if len(normal_files) < N_TEST_NORMAL + N_VAL_NORMAL + N_TRAIN_NORMAL:
    print("[error] not enough normal images for the configured split")
    sys.exit(1)

assign = []  # (path, label, split)
for i, p in enumerate(defect_files):
    s = "test" if i < N_TEST_DEFECT else ("val" if i < N_TEST_DEFECT + N_VAL_DEFECT else "train")
    assign.append((p, 1, s))
for i, p in enumerate(normal_files):
    if i < N_TEST_NORMAL:
        s = "test"
    elif i < N_TEST_NORMAL + N_VAL_NORMAL:
        s = "val"
    elif i < N_TEST_NORMAL + N_VAL_NORMAL + N_TRAIN_NORMAL:
        s = "train"
    else:
        s = "unused"
    assign.append((p, 0, s))


# ---------------- helpers ----------------
def resize_to(img, size_wh, interp):
    if (img.shape[1], img.shape[0]) == size_wh:
        return img
    return cv2.resize(img, size_wh, interpolation=interp)


def mask_to_boxes(mask_bin):
    h, w = mask_bin.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BOX_DILATE_PX, BOX_DILATE_PX))
    m = cv2.dilate(mask_bin, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    xyxy = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < BOX_MIN_AREA:
            continue
        xyxy.append([x, y, x + bw, y + bh])
    # merge boxes that overlap or sit inside each other into one box (repeat until stable),
    # boxes that are apart (for example two separate bent pins) stay separate
    merged = True
    while merged:
        merged = False
        for i in range(len(xyxy)):
            for j in range(i + 1, len(xyxy)):
                a, b = xyxy[i], xyxy[j]
                if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                    xyxy[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del xyxy[j]
                    merged = True
                    break
            if merged:
                break
    return [[(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h] for x0, y0, x1, y1 in xyxy]


# ---------------- build dataset ----------------
rows = []
n_fail = 0
for p, label, split in assign:
    image_id = f"{CATEGORY}_{'defect' if label else 'normal'}_{p.stem}"
    row = dict(image_id=image_id, split=split, label=label, path="", src_path=str(p),
               width=0, height=0, n_boxes=0, boxes="[]", defect_area_frac=0.0)
    if split == "unused":
        rows.append(row)
        continue
    try:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imread returned None")
        h0, w0 = img.shape[:2]
        s = min(1.0, MAX_SIDE / max(h0, w0))
        size = (max(1, round(w0 * s)), max(1, round(h0 * s)))
        img = resize_to(img, size, cv2.INTER_AREA)
        W, H = size

        boxes, area_frac = [], 0.0
        if label == 1:
            m = cv2.imread(str(find_mask(p)), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise ValueError("mask unreadable")
            m = resize_to(m, size, cv2.INTER_NEAREST)
            # VisA masks store the defect type as pixel value (1, 3, 4 ...), not 255, so use > 0
            mb = np.ascontiguousarray((m > 0).astype(np.uint8))
            area_frac = float(mb.sum()) / (W * H)
            boxes = mask_to_boxes(mb)
            if not boxes and mb.sum() > 0:
                ys, xs = np.nonzero(mb)
                x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                boxes = [[(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]]
            (MASK_DIR / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(MASK_DIR / split / f"{image_id}.png"), mb * 255)

        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        out_img = YOLO_DIR / "images" / split / f"{image_id}.jpg"
        cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(YOLO_DIR / "labels" / split / f"{image_id}.txt", "w") as f:
            for b in boxes:
                f.write("0 " + " ".join(f"{v:.6f}" for v in b) + "\n")

        row.update(path=str(out_img.relative_to(OUT_DIR)), width=W, height=H,
                   n_boxes=len(boxes), boxes=json.dumps(boxes), defect_area_frac=area_frac)
    except Exception as e:
        n_fail += 1
        row["split"] = "failed"
        print(f"[warn] failed on {p.name}: {e}")
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "split.csv", index=False)

# YOLO config
with open(YOLO_DIR / "data.yaml", "w") as f:
    f.write(f"path: {YOLO_DIR.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: defect\n")

# metadata + license note
meta = dict(category=CATEGORY, seed=SEED, max_side=MAX_SIDE, source_url=VISA_URL,
            counts=df.groupby(["split", "label"]).size().rename("n").reset_index().to_dict("records"),
            failed=n_fail,
            license_note=("VisA: CC BY 4.0 per the authors' repo (amazon-science/spot-diff) and the AWS "
                          "Open Data registry. Some third-party docs list CC BY-NC-SA 4.0, confirm with "
                          "the company before publishing. Cite Zou et al., ECCV 2022, arXiv:2207.14315."))
with open(OUT_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# ---------------- sanity checks ----------------
print("\n[summary] images per split and label (0 normal, 1 defect)")
print(df[df.split != "unused"].groupby(["split", "label"]).size().unstack(fill_value=0))
d = df[(df.label == 1) & (df.split.isin(["train", "val", "test"]))]
if len(d):
    print(f"[summary] defect area fraction: median={d.defect_area_frac.median():.5f} "
          f"min={d.defect_area_frac.min():.5f} max={d.defect_area_frac.max():.5f}")
    print(f"[summary] boxes per defect image: mean={d.n_boxes.mean():.2f}, images with 0 boxes={(d.n_boxes == 0).sum()}")
print(f"[summary] failed images: {n_fail}")

# preview grid with boxes drawn (open preview.jpg and CHECK that boxes sit on real defects)
try:
    prev = d.sample(n=min(8, len(d)), random_state=SEED)
    tiles = []
    for _, r in prev.iterrows():
        im = cv2.imread(str(OUT_DIR / r.path))
        H, W = im.shape[:2]
        for xc, yc, bw, bh in json.loads(r.boxes):
            x0, y0 = int((xc - bw / 2) * W), int((yc - bh / 2) * H)
            x1, y1 = int((xc + bw / 2) * W), int((yc + bh / 2) * H)
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), max(2, W // 300))
        tiles.append(cv2.resize(im, (480, int(480 * H / W))))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(OUT_DIR / "preview.jpg"), grid)
    print(f"[summary] preview saved to {OUT_DIR / 'preview.jpg'}")
except Exception as e:
    print(f"[warn] preview failed: {e}")

# zip for reuse as a private Kaggle dataset (so later notebooks skip the download)
try:
    import zipfile
    zip_path = OUT_DIR / f"visa_{CATEGORY}_bench.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for top in ["yolo", "masks"]:
            for fp in (OUT_DIR / top).rglob("*"):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(OUT_DIR))
        for name in ["split.csv", "meta.json", "preview.jpg"]:
            if (OUT_DIR / name).exists():
                zf.write(OUT_DIR / name, name)
    print(f"[done] zip saved: {zip_path}")
except Exception as e:
    print(f"[warn] zip failed: {e}")

print("[done] data prep finished")
