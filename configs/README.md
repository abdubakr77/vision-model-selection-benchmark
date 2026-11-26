# configs

One yaml file per model with its training settings, read by the model notebook. Settings shared by every model stay fixed in the data prep and the notebooks: seed 42, the split in `bench/split.csv`, and thresholds chosen on val only.

| File | Used by | What it holds |
|---|---|---|
| `yolo26s.yaml` | `notebooks/manufacturing/03_yolo.ipynb` | YOLO26-S training settings: image size 1024, up to 50 epochs with early stopping after 7, batch 32 on two GPUs with gradients summed to 64, AdamW 0.0005 to 0.00005, light built-in augmentation (brightness, small shift and zoom, no flips). The exact settings of each run are saved in its run folder as `args.yaml` |
