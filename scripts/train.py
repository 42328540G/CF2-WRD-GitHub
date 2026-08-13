#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from cf2wrd.data import PairedImageTransform, WheatRustDataset
from cf2wrd.model import build_model
from cf2wrd.preprocessing import LabKMeansLesionPreprocessor
from cf2wrd.selector import fit_local_complementary_selector
from cf2wrd.utils import (
    classification_metrics,
    load_config,
    resolve_device,
    set_seed,
)


def make_transform(cfg, train):
    p = cfg["preprocessing"]
    a = cfg["augmentation"]
    pre = None
    if p.get("enabled", True):
        pre = LabKMeansLesionPreprocessor(
            k=p.get("k", 3),
            morphology_kernel=p.get("morphology_kernel", 3),
            output=p.get("local_output", "masked_rgb"),
            outside_scale=p.get("outside_scale", 0.15),
            min_cluster_area_ratio=p.get("min_cluster_area_ratio", 0.002),
            a_weight=p.get("a_weight", 0.25),
            attempts=p.get("kmeans_attempts", 3),
            seed=cfg.get("seed", 42),
        )
    return PairedImageTransform(
        input_size=cfg["data"].get("input_size", 224),
        train=train,
        lesion_preprocessor=pre,
        rotation_degrees=a.get("rotation_degrees", 15),
        horizontal_flip_p=a.get("horizontal_flip_p", 0.5),
        gaussian_blur_p=a.get("gaussian_blur_p", 0.3),
        gaussian_kernel=a.get("gaussian_kernel", 3),
        brightness_delta=a.get("brightness_delta", 0.2),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    losses = []
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        gx = batch["global_image"].to(device)
        lx = batch["local_image"].to(device)
        y = batch["label"].to(device)
        logits = model(gx, lx)
        losses.append(float(criterion(logits, y).item()))
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = sum(losses) / max(len(losses), 1)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cf2_wrd.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "auto"))

    root = cfg["data"].get("root", ".")
    train_ds = WheatRustDataset(
        cfg["data"]["train_csv"], root=root, transform=make_transform(cfg, train=True)
    )
    # Deterministic training view for fitting Pearson correlations.
    selector_ds = WheatRustDataset(
        cfg["data"]["train_csv"], root=root, transform=make_transform(cfg, train=False)
    )
    val_ds = WheatRustDataset(
        cfg["data"]["val_csv"], root=root, transform=make_transform(cfg, train=False)
    )

    tr_cfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=tr_cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )
    selector_loader = DataLoader(
        selector_ds,
        batch_size=tr_cfg.get("selector_batch_size", 64),
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tr_cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )

    # Build encoder first, fit training-only selector, then instantiate fusion.
    model = build_model(cfg).to(device)
    sel = fit_local_complementary_selector(
        model.encoder,
        selector_loader,
        device=device,
        tau=cfg["model"].get("correlation_threshold", 0.7),
    )
    model.set_selected_local_indices(sel.selected_local_indices)
    model = model.to(device)

    print(
        f"Selected {len(sel.selected_local_indices)}/{cfg['model'].get('local_dim', 1024)} "
        f"local dims; mean |r| all={sel.mean_abs_correlation_all:.4f}; "
        f"mean |r| selected={sel.mean_abs_correlation_selected:.4f}"
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=tr_cfg.get("label_smoothing", 0.1)
    )
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tr_cfg.get("lr", 1e-4),
        weight_decay=tr_cfg.get("weight_decay", 0.01),
    )
    scheduler = StepLR(
        optimizer,
        step_size=tr_cfg.get("lr_step_size", 10),
        gamma=tr_cfg.get("lr_gamma", 0.9),
    )

    ckpt_dir = Path(tr_cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / tr_cfg.get("best_name", "cf2_wrd_best.pt")
    best_acc = -1.0

    for epoch in range(1, tr_cfg.get("epochs", 60) + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}")
        for batch in pbar:
            gx = batch["global_image"].to(device, non_blocking=True)
            lx = batch["local_image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(gx, lx)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        print(
            f"epoch={epoch} train_loss={running/max(len(train_loader),1):.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "selected_local_indices": model.selected_local_indices.cpu().tolist(),
                    "selector_mean_abs_correlation_all": sel.mean_abs_correlation_all,
                    "selector_mean_abs_correlation_selected": sel.mean_abs_correlation_selected,
                    "config": cfg,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
