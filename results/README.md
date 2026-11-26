# results

- `per_image/`: one CSV per model and dataset. Columns: image_id, model, prediction, score, latency_ms, vram_mb, raw_output, parse_ok, tokens_in, tokens_out.
- `tables/`: aggregated comparison tables.
- `yolo/<dataset>/<run name>/`: Ultralytics run folders (curves, confusion matrix, sample batches, `results.csv`, `args.yaml`). Checkpoints (`*.pt`) are git-ignored.
- `figures/<scenario>/`: charts and image grids used in the README, docs and blog. File names start with the dataset, for example `figures/manufacturing/pcb1_orientation_defect_upright.png`.

Cached cloud API responses stay local (git-ignored) so runs are reproducible without paying twice.
