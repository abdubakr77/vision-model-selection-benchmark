# Methodology

## Fairness rules

1. Every model family solves the same task per dataset, scored with the same metric. No comparing YOLO boxes against free text.
2. One fixed split per dataset (seed 42), same preprocessing, logged per image.
3. Thresholds and prompts are tuned on validation data only, never on test.
4. Accuracy is one of eight dimensions. The result is a decision framework.
5. Cloud models run on small test subsets, every response is cached, and token usage is logged.
6. VLM outputs are structured JSON. Parse failures are counted and reported.

## Task framing

| Dataset | Shared task |
|---|---|
| Manufacturing | image-level normal / defective |
| Documents | field extraction or question answering |
| Retail | product classification or counting |

## Data preparation rules

- Fixes found in EDA are applied once, in the data prep notebook, so every model sees the same images.
- Augmentation is applied to train only, offline, with a fixed seed. Val and test are never augmented. Built-in augmentation of a model library is kept light so it does not stack with the offline copies: for YOLO only brightness, a small shift and a small zoom, with no flips or rotation (board orientation is fixed).
- Every image is stored in color and in grayscale. Supervised models are trained on grayscale first, then retrained on color for comparison; VLMs and cloud models get color.

## Metrics

- Quality: precision, recall, F1, accuracy, plus AUROC when a score exists.
- Errors: false positives and false negatives reported separately (cost of each differs per use case).
- Latency: per-image milliseconds, median and p95, after warm-up.
- GPU memory: peak allocated during inference.
- Model size: parameters and size on disk.
- Cost: per 1,000 images. Local = hardware time; cloud = token usage times public price (not what the free tier charges).
- Explainability: what the model can show or say to justify a result (boxes, scores, text).
- Uncertainty: small test sets, so report bootstrap confidence intervals.

## Per-image log schema

image_id, model, prediction, score, latency_ms, vram_mb, raw_output, parse_ok, tokens_in, tokens_out.

## Reproducibility

seed_everything(42), fixed split files, pinned requirements, Kaggle T4 as the reference environment.
