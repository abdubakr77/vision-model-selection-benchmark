# configs

One yaml file per model with its training settings, read by the model notebook. Settings shared by every model stay fixed in the data prep and the notebooks: seed 42, the split in `bench/split.csv`, and thresholds chosen on val only.

| File | Used by | What it holds |
|---|---|---|
| `yolo26s.yaml` | `notebooks/manufacturing/03_yolo.ipynb` | YOLO26-S training settings: image size 1024, up to 50 epochs with early stopping after 7, batch 32 on two GPUs with gradients summed to 64, AdamW 0.0005 to 0.00005, light built-in augmentation (brightness, small shift and zoom, no flips). The exact settings of each run are saved in its run folder as `args.yaml` |
| `swin_b.yaml` | `notebooks/manufacturing/04_vit_swin.ipynb` | Swin-B classifier settings: input 640 x 1024 (the board crop), up to 20 epochs with early stopping after 5, batch 4 per GPU on two GPUs with gradients summed to 16, AdamW 0.0001 with warmup and cosine, no online augmentation |
