# Vision Model Selection Benchmark

A practical, reproducible benchmark that answers one question: **when should a company use YOLO, a Vision Transformer, an open-source VLM, a cloud vision model, or a private/local model?**

The same task, split, preprocessing and metrics are applied to every model type across three enterprise scenarios (manufacturing, documents, retail). The output is a decision framework, not a leaderboard.

> Status: Phase 1 in progress (one dataset, three models). See [docs/CHECKLIST.md](docs/CHECKLIST.md).

Built as part of a remote internship at GenAI Protos. Author: Abdullah Bakr.

## What is measured

Eight dimensions per model: task quality (accuracy, precision, recall, F1), latency, GPU memory, model size, false positives, false negatives, cost per 1,000 images, and explainability. For VLMs, JSON parse failures are reported as a result in their own right.

## Scenarios

| Scenario | Dataset | Task | Status |
|---|---|---|---|
| Manufacturing defect detection | VisA (`pcb1` first) | image-level normal / defective | Phase 1 |
| Document understanding | to be selected | field extraction or QA | Phase 3 |
| Retail / product recognition | to be selected | classification or counting | Phase 3 |

## Model families

YOLO (detection), Vision Transformer / Swin (classifier), open-source VLM, Gemma-family multimodal model, cloud-hosted vision model, local/private model. Choices and rationale: [docs/model-selection.md](docs/model-selection.md).

## Repository layout

```
.
├── README.md
├── LICENSE                  Apache-2.0 (code only)
├── requirements.txt
├── configs/                 benchmark settings (seed, split sizes, thresholds)
├── datasets/                one folder per scenario: README + preparation script
│   ├── manufacturing_visa/
│   ├── documents/
│   └── retail/
├── src/vision_benchmark/    reusable code
│   ├── models/              one wrapper per model, same interface
│   ├── evaluation/          metrics, timing, memory, cost
│   └── utils/               seeding, IO, logging
├── notebooks/               Kaggle / Colab runners
├── scripts/                 command-line entry points
├── results/                 per-image logs and result tables
├── demo/                    upload-and-compare demo app
└── docs/                    methodology, findings, decision framework, blog, posts, diagrams
```

## Getting started

Setup instructions are added once the Phase 1 pipeline runs end to end. Data is never stored in the repo; each `datasets/*/README.md` explains how to download it.

## Licenses and attribution

- Code in this repo: Apache-2.0.
- VisA dataset: CC BY 4.0 according to its authors (confirm before publishing). Cite Zou et al., "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation", ECCV 2022.
- Ultralytics YOLO is AGPL-3.0 (an enterprise license exists). Review before using it in client work.
- Gemma models are under Google's Gemma terms and must be accepted by each user on Hugging Face.
