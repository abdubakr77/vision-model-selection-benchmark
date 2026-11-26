# notebooks

One folder per scenario (`manufacturing/`, `documents/`, `retail/`). Each folder holds every notebook of that dataset, from EDA to model runs. Notebooks are thin wrappers around code in `src/vision_benchmark` and write their outputs to `results/`.

Notebooks import helpers from the repo package. Install it once from the repo root with `pip install -e .`; after that a kernel restart picks up any change in `src/`.

## Order of work

Per dataset (one dataset at a time, Phase 1 = manufacturing):

| Notebook | What it does | Phase |
|---|---|---|
| 01_eda | explore the raw data, find problems and edge cases; never changes the data, ends with a findings table | 1 |
| 02_data_prep | apply the fixes from the EDA, build `bench/` and the fixed `split.csv` | 1 |
| 03_yolo | YOLO detector, trained on gray then on color, image-level results on val and test | 1 |
| 04_vit_swin | Swin-B classifier, trained on gray then on color, image-level results on val and test | 1 |
| 05_vlm_open | open-source VLM | 1 |
| 06_gemma | Gemma multimodal | 1 or 2 |
| 07_cloud_api | cloud API model | 2 |
| 08_results | tables and charts | 1 onward |

`manufacturing/` has all of them; `documents/` and `retail/` get their model notebooks in Phase 3.

Training runs on Kaggle (T4 x2) with a self-contained copy of the notebook; the run folder is then downloaded into `results/` and the repo notebook reads it from there.

## Rules

- Every model reads the same `split.csv` and logs one row per test image.
- Thresholds and prompts are tuned on val only.
- Secrets (HF token, API keys) live in Kaggle Secrets, never in a notebook.
