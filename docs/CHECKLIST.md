# Deliverables checklist

Status: not started / in progress / done

| Deliverable | Status |
|---|---|
| Dataset research and selected datasets | in progress (manufacturing chosen) |
| Model selection and rationale | not started |
| Reusable benchmark / evaluation code | in progress (installable package, box, plotting, orientation, contrast and crop helpers, image-level and detection metrics) |
| Results comparison table | in progress (YOLO gray vs color) |
| Accuracy, latency, cost, infrastructure comparison | not started |
| Failure-analysis examples | not started |
| Cloud vs private/local comparison | not started |
| Model-selection decision framework | not started |
| Interactive demo | not started |
| Architecture / solution diagrams | not started |
| Technical findings document | not started |
| Technical blog | not started |
| 3 to 5 LinkedIn posts | not started |
| Short demo video | not started |
| Clean GitHub repo with setup instructions | in progress (structure, install and data prep steps) |

## Phases

- Phase 0: questions answered. Done.
- Phase 1: one dataset (VisA pcb1), YOLO + Swin/ViT + one small open VLM, metrics, latency, VRAM. In progress: EDA on raw pcb1 done, `02_data_prep` done (orientation fixed, board crop, stratified split, gray and color copies, train-only augmentation), YOLO26-S trained on gray and on color. Next: Swin/ViT and one small VLM.
- Phase 2: cloud API model(s), cost and latency tracking.
- Phase 3: documents and retail datasets, edge cases, failure analysis.
- Phase 4: tables, charts, decision framework, demo, blog, posts, video.
