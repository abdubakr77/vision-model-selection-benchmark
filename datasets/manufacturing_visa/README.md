# Manufacturing: VisA (pcb1)

**Source:** https://github.com/amazon-science/spot-diff (download via AWS Open Data: `s3://amazon-visual-anomaly/VisA_20220922.tar`, about 2 GB).
**Size:** 12 object types, 10,821 images (9,621 normal, 1,200 anomalous), about 100 anomalous images per type. Image and pixel-level (mask) annotations.
**Subset used first:** `pcb1`, an HC-SR04 ultrasonic sensor board (1004 normal, 100 defective). Defects are small, which makes it a hard and realistic test.

## Task (shared by every model)

Image-level classification: `normal` or `defective`.

- YOLO: image is defective if any box exceeds a confidence threshold.
- ViT / Swin: defect probability with a threshold.
- VLM: JSON answer `{"defective": true|false, "reason": "..."}`.

Thresholds are tuned on the validation split only.

## Getting the data

Download the VisA tar and extract it into this folder (git-ignored), so the files sit at:

```
datasets/manufacturing_visa/pcb1/image_anno.csv
datasets/manufacturing_visa/pcb1/Data/Images/{Normal,Anomaly}/
datasets/manufacturing_visa/pcb1/Data/Masks/Anomaly/
```

`image_anno.csv` has one row per image: image path, defect type (`normal`, `bent`, `melt`, `scratch`, `missing`, or several joined by commas) and mask path.

## Preparation

Run [notebooks/manufacturing/02_data_prep.ipynb](../../notebooks/manufacturing/02_data_prep.ipynb) top to bottom (about 1 minute locally). It reads the raw files above and rebuilds `bench/` at the repo root:

```
bench/
  split.csv, meta.json
  color/data.yaml, color/images/<split>/, color/labels/<split>/
  gray/data.yaml,  gray/images/<split>/,  gray/labels/<split>/
  masks/<split>/          defect images only, 0 or 255
```

Steps, in order:

1. Measure which defect images have a small box (under 32 x 32 px at 1024 x 780).
2. Split with seed 42 (table below). Defect images are split separately with and without a small box, in the same proportion.
3. Detect the board orientation of every used image and turn `rotated_180` images upright, together with their masks.
4. Resize to 1024 x 780, derive boxes from the masks, write a color and a gray copy with YOLO label files.
5. Train only: 3 augmented copies of every defect image, a 50% chance of one copy for every normal image. Each copy gets a random safe zoom in (70% chance, up to about 1.4x, never cutting the board, pins or defects) and random brightness and contrast.
6. Write `split.csv`, one `data.yaml` per mode, `meta.json`, then check the output.

## Split (seed 42)

| Split | Defective | Normal | Defective with a small box |
|---|---|---|---|
| train | 50 (+150 augmented) | 150 (+76 augmented) | 11 |
| val | 20 | 100 | 5 |
| test | 30 | 300 | 7 |

The remaining 454 normal images are not used. Val and test are never augmented.

Why these sizes: only 100 defective images exist, so every defect in test is one less for training. Test uses a near-realistic ratio (1:10) so false alarms can be measured with useful resolution. Val has a similar ratio so thresholds tuned on val transfer to test. The official VisA split files (`split_csv/`) were not used: `1cls` has no defects in train, and the `2cls` files have no validation split.

## Data notes

- **Masks look black but are not empty.** The pixel value is the defect type id (1 bent, 2 melt, 3 scratch, 4 missing), not 255. Any threshold like `> 127` returns an empty mask. Use `> 0`.
- Raw images and masks are 1404 x 1070.
- Boxes are derived from masks (dilate 9 px, connected components, drop specks under 15 px). Boxes that overlap or sit inside each other are merged into one; separate defects (for example two bent pins) keep separate boxes.
- Paths in `split.csv` use `/`, so `bench/` works on Linux (Kaggle) unchanged. The YOLO `data.yaml` files have no `path` key, so YOLO uses the folder of the yaml.

## EDA findings

From [notebooks/manufacturing/01_eda.ipynb](../../notebooks/manufacturing/01_eda.ipynb), on the raw data. Fixes are applied in `02_data_prep`.

| Finding | Evidence |
|---|---|
| Four defect types | melt 54, scratch 21, missing 20, bent 15 images; 9 images have more than one type |
| Defects are small | median defect area 0.2% of the image, smallest box 9 x 9 px |
| Small defects are a group of their own | 26 of 100 defect images have a box under 32 x 32 px at 1024 wide; all of them contain `melt` |
| Board orientation leaks the label | 790 of 1004 normal images are rotated 180 degrees, 96 of 100 defect images are upright |
| Color adds little | brightness difference beats color difference for 93% of defects |
| No duplicate boxes | 0 inside an image, 0 across images |

The small-box count is 26 on raw boxes scaled to 1024 wide and 23 on boxes measured after the resize; the split uses the second.

## License and citation

CC BY 4.0 per the authors; some third-party docs list CC BY-NC-SA 4.0, so confirm with the company before publishing. Cite Zou et al., ECCV 2022 (arXiv:2207.14315).
