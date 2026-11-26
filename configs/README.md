# configs

Benchmark settings shared by all models: seed (42), split sizes, image size, thresholds source (validation split only), and model identifiers. One YAML file per experiment.

| File | Used by | What it holds |
|---|---|---|
| `yolo26s.yaml` | `notebooks/models/03_yolo.ipynb` | YOLO26-S training settings: image size 1024, up to 50 epochs with early stopping after 7, batch 32 on two GPUs with gradients summed to 64, AdamW 0.0005 to 0.00005, light built-in augmentation (brightness, small shift and zoom, no flips). The exact settings of each run are saved in its run folder as `args.yaml` |
