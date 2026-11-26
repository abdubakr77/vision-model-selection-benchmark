"""Image-level classifier (Swin / ViT from timm): dataset, model, training loop and prediction."""

import math
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SplitDataset(torch.utils.data.Dataset):
    """Images of one split from bench/split.csv, returned as (normalized tensor, label).

    mode "gray" reads path_gray, "color" reads path. size is (height, width).
    No augmentation: train was already augmented offline in data prep.
    """

    def __init__(self, bench_dir, split, mode, size):
        df = pd.read_csv(os.path.join(bench_dir, "split.csv"))
        self.df = df[df["split"] == split].reset_index(drop=True)
        column = "path_gray" if mode == "gray" else "path"
        self.paths = [os.path.join(bench_dir, p) for p in self.df[column]]
        self.labels = self.df["label"].to_numpy(np.float32)
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)  # a gray jpg comes back as 3 equal channels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.tensor(self.labels[i])


def seed_everything(seed):
    """Same seed for python, numpy and torch, and deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, batch, shuffle, workers, seed=42):
    """DataLoader with a seeded shuffle so the image order is the same on every run."""
    return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                                       pin_memory=True, generator=torch.Generator().manual_seed(seed))


def build_classifier(name, size, drop_path_rate=0.1, pretrained=True):
    """timm model with one output (defect logit) at input size (height, width).

    Swin works on windows, so it accepts sizes other than the one it was pretrained on
    (both sides must be multiples of 32).
    """
    return timm.create_model(name, pretrained=pretrained, num_classes=1, img_size=size,
                             drop_path_rate=drop_path_rate)


def warmup_cosine(warmup_steps, total_steps):
    """LR factor: linear rise during warmup, then cosine decay to 0."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return factor


@torch.no_grad()
def predict(model, loader, device="cuda"):
    """Defect probability for every image in the loader. Returns (scores, labels, mean BCE loss)."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x.to(device, non_blocking=True)).squeeze(1)
        logits.append(out.float().cpu())
        labels.append(y)
    logits, labels = torch.cat(logits), torch.cat(labels)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    return torch.sigmoid(logits).numpy(), labels.numpy(), loss


def better_epoch(auroc, loss, best_auroc, best_loss, peak_auroc, auroc_floor, auroc_tol, min_loss_drop,
                 max_loss_rise):
    """Why this epoch beats the saved best epoch, or None if it does not.

    best_auroc, best_loss: val AUROC and val loss of the saved epoch. peak_auroc: highest val AUROC of any
    earlier epoch.
    1. While AUROC is under auroc_floor, ranking still has room to grow: only a higher AUROC counts.
    2. The first epoch at or above the floor counts.
    3. Once AUROC is above the floor, small changes are noise (with 20 val defects and 100 normals one pair
       moves AUROC by 0.0005). The epoch counts if:
       - AUROC jumps by auroc_tol or more and val loss is at most max_loss_rise (a fraction) over the saved best,
         so an overconfident model is not taken for a better one, or
       - AUROC stays within auroc_tol of the peak and val loss is at least min_loss_drop (a fraction) under the
         saved best. A val loss that falls only a little does not count.
    """
    if auroc < auroc_floor:
        if best_auroc >= auroc_floor:
            return None
        if auroc > best_auroc:
            return "AUROC up"
        if auroc == best_auroc and loss < best_loss:
            return "same AUROC, lower val loss"
        return None
    if best_auroc < auroc_floor:
        return f"AUROC reached {auroc_floor}"
    if auroc >= best_auroc + auroc_tol and loss <= best_loss * (1 + max_loss_rise):
        return "AUROC up"
    if auroc >= peak_auroc - auroc_tol and loss <= best_loss * (1 - min_loss_drop):
        return f"val loss down {100 * (1 - loss / best_loss):.1f}%, AUROC held"
    return None


def fit(model, train_loader, val_loader, epochs, lr, weight_decay, accum_steps, warmup_epochs, run_dir,
        patience=5, auroc_floor=0.98, auroc_tol=0.005, min_loss_drop=0.05, max_loss_rise=0.2, device="cuda"):
    """Train with AdamW, warmup + cosine LR, fp16 and gradient accumulation.

    One weight update every accum_steps batches. After each epoch the model is scored on val and
    better_epoch decides if it is the new best (val AUROC first, then val loss once AUROC is high);
    the best epoch is saved as run_dir/best.pt. Training stops early after `patience` epochs
    without a new best (None = never).
    Works with a plain model or one wrapped in torch.nn.DataParallel (several GPUs);
    best.pt always holds the plain model weights, so it loads on one GPU.
    Prints one line per epoch and saves history.csv. Returns the history DataFrame.
    """
    os.makedirs(run_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / accum_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(updates_per_epoch * warmup_epochs, updates_per_epoch * epochs))
    scaler = torch.amp.GradScaler()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"monitor: val AUROC until it reaches {auroc_floor}. After that a new best needs either AUROC up by "
          f"{auroc_tol} with val loss at most {100 * max_loss_rise:.0f}% higher, or AUROC within {auroc_tol} of "
          f"its peak with val loss at least {100 * min_loss_drop:.0f}% lower than the best epoch")
    print(f"early stopping: patience {patience}" if patience else "early stopping: off")

    history = []
    best_auroc, best_loss, best_epoch, peak_auroc, wait = -1.0, float("inf"), 0, -1.0, 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, seen = 0.0, 0
        for i, (x, y) in tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"epoch {epoch}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x).squeeze(1).float(), y)
            scaler.scale(loss / accum_steps).backward()
            if i % accum_steps == 0 or i == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            total += loss.item() * len(y)
            seen += len(y)

        scores, labels, val_loss = predict(model, val_loader, device)
        auroc = roc_auc_score(labels, scores)
        reason = better_epoch(auroc, val_loss, best_auroc, best_loss, peak_auroc, auroc_floor, auroc_tol,
                              min_loss_drop, max_loss_rise)
        peak_auroc = max(peak_auroc, auroc)
        if reason:
            best_auroc, best_loss, best_epoch, wait = auroc, val_loss, epoch, 0
            plain = model.module if isinstance(model, torch.nn.DataParallel) else model  # drop the "module." prefix
            torch.save(plain.state_dict(), os.path.join(run_dir, "best.pt"))
            status = f"best ({reason})"
        else:
            wait += 1
            status = f"no improvement {wait}/{patience}" if patience else "no improvement"

        row = {"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss, "val_auroc": auroc,
               "lr": optimizer.param_groups[0]["lr"], "time_s": time.time() - start, "best": reason or ""}
        history.append(row)
        print(f"epoch {epoch:2d}/{epochs} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val AUROC {auroc:.4f} | lr {row['lr']:.2e} | {row['time_s']:.0f}s | {status}")

        if patience and wait >= patience:
            print(f"early stop: no new best for {patience} epochs")
            break

    print(f"best epoch {best_epoch}: val AUROC {best_auroc:.4f}, val loss {best_loss:.4f}")
    history = pd.DataFrame(history)
    history.to_csv(os.path.join(run_dir, "history.csv"), index=False)
    return history
