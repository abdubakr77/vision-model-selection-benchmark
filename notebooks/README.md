# notebooks

Kaggle / Colab runners. Each notebook is a thin wrapper around code in `src/vision_benchmark` and writes its outputs to `results/`.

## Order of work

Per dataset (one dataset at a time, Phase 1 = manufacturing):

1. `01_eda`: explore the data, find problems and edge cases. It never changes the data; it ends with a findings table.
2. `02_data_prep`: apply the fixes from the EDA and build the canonical dataset and the fixed `split.csv`.

Notebooks import helpers from the repo package. Install it once from the repo root with `pip install -e .`; after that a kernel restart picks up any change in `src/`.

Models (shared across datasets, dataset name is a parameter):

| Notebook | Model | Phase |
|---|---|---|
| models/03_yolo | YOLO | 1 |
| models/04_vit_swin | ViT / Swin classifier | 1 |
| models/05_vlm_open | open-source VLM | 1 |
| models/06_gemma | Gemma multimodal | 1 or 2 |
| models/07_cloud_api | cloud API model | 2 |
| models/08_results | tables and charts | 1 onward |

## Rules

- Every model reads the same `split.csv` and logs one row per test image.
- Thresholds and prompts are tuned on val only.
- Secrets (HF token, API keys) live in Kaggle Secrets, never in a notebook.
