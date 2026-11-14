# Manufacturing: VisA (pcb1)

**Source:** https://github.com/amazon-science/spot-diff (download via AWS Open Data: `s3://amazon-visual-anomaly/VisA_20220922.tar`).
**Size:** 12 object types, 10,821 images (9,621 normal, 1,200 anomalous), about 100 anomalous images per type. Image and pixel-level (mask) annotations.
**Subset used first:** `pcb1` (about 904 normal, 100 defective). Defects are small, which makes it a hard and realistic test.

## Task (shared by every model)

Image-level classification: `normal` or `defective`.

- YOLO: image is defective if any box exceeds a confidence threshold.
- ViT / Swin: defect probability with a threshold.
- VLM: JSON answer `{"defective": true|false, "reason": "..."}`.

Thresholds are tuned on the validation split only.

## Split (seed 42)

| Split | Defective | Normal |
|---|---|---|
| train | 55 | 165 |
| val | 20 | 20 |
| test | 25 | 25 |

Remaining normal images are marked `unused` in `split.csv`.

## Preparation

```
python prepare_data.py [input_dir] [output_dir]
```

Downloads the tar if needed (Kaggle: turn Internet on), extracts one category, resizes to max side 1024, derives YOLO boxes from the masks, and writes `split.csv`, `meta.json`, YOLO `data.yaml` and a `preview.jpg` for visual checking.

## License and citation

CC BY 4.0 per the authors; some third-party docs list CC BY-NC-SA 4.0, so confirm with the company before publishing. Cite Zou et al., ECCV 2022 (arXiv:2207.14315).
