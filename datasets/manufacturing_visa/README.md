# Manufacturing: VisA (pcb1)

**Source:** https://github.com/amazon-science/spot-diff (download via AWS Open Data: `s3://amazon-visual-anomaly/VisA_20220922.tar`).
**Size:** 12 object types, 10,821 images (9,621 normal, 1,200 anomalous), about 100 anomalous images per type. Image and pixel-level (mask) annotations.
**Subset used first:** `pcb1` (1004 normal, 100 defective, counted on disk). Defects are small, which makes it a hard and realistic test.

## Task (shared by every model)

Image-level classification: `normal` or `defective`.

- YOLO: image is defective if any box exceeds a confidence threshold.
- ViT / Swin: defect probability with a threshold.
- VLM: JSON answer `{"defective": true|false, "reason": "..."}`.

Thresholds are tuned on the validation split only.

## Split (seed 42)

| Split | Defective | Normal |
|---|---|---|
| train | 50 | 150 |
| val | 20 | 100 |
| test | 30 | 300 |

The remaining 454 normal images are marked `unused` in `split.csv`.

Why these sizes: only 100 defective images exist, so every defect in test is one less for training. Test uses a near-realistic ratio (1:10) so false alarms can be measured with useful resolution. Val has a similar ratio so thresholds tuned on val transfer to test. Train is kept near 1:3. The official VisA split files (`split_csv/`) were not used: `1cls` has no defects in train, and the `2cls` files have no validation split.

## Data notes

- **Masks look black but are not empty.** VisA masks store the defect type as the pixel value (0, 1, 3, 4 ...), not 255. Any threshold like `> 127` returns an empty mask. Use `> 0`.
- `pcb1` is an HC-SR04 ultrasonic sensor board. Defect size varies a lot: a bent pin covers about 0.1% of the image, a damaged sensor part about 10%. This spread is useful for failure analysis.
- Images are 1404 x 1070 and masks have the same size.
- Boxes are derived from masks (dilate 9 px, connected components, drop specks under 15 px). Boxes that overlap or sit inside each other are merged into one; separate defects (for example two bent pins) keep separate boxes.

## Preparation

Local layout used during development (git-ignored): the VisA download is extracted into this folder, so `pcb1` sits at `datasets/manufacturing_visa/pcb1/Data/Images/{Normal,Anomaly}` and `.../Data/Masks/Anomaly`.

From the repo root:

```
python datasets/manufacturing_visa/prepare_data.py datasets/manufacturing_visa bench
```

Arguments are `[input_dir] [output_dir]`. If `pcb1` is not found under the input folder, the script downloads the tar (Kaggle: turn Internet on) and extracts only that category. It resizes to max side 1024, derives YOLO boxes from the masks, and writes `split.csv`, `meta.json`, YOLO `data.yaml` and a `preview.jpg` for visual checking. A full local run takes about 15 seconds.

## EDA findings

From [notebooks/manufacturing/01_eda.ipynb](../../notebooks/manufacturing/01_eda.ipynb). Fixes are applied in `02_data_prep`.

| Finding | Evidence |
|---|---|
| Defects are small | median defect area 0.2% of the image, smallest box 9 x 9 px at 1024 x 780 |
| Small defects are unevenly split | 23 of 100 defect images have a box under 32 x 32 px: 17 train, 3 val, 3 test |
| Board orientation leaks the label | 439 of 550 normal images are rotated 180 degrees, 96 of 100 defect images are upright |
| Color adds little | brightness difference beats color difference for 92% of defects |
| No duplicate boxes | 0 inside an image, 0 across images |

## License and citation

CC BY 4.0 per the authors; some third-party docs list CC BY-NC-SA 4.0, so confirm with the company before publishing. Cite Zou et al., ECCV 2022 (arXiv:2207.14315).
