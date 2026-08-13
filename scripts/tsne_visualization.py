#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from cf2wrd.data import WheatRustDataset
from cf2wrd.model import build_model
from cf2wrd.utils import load_checkpoint, load_config, resolve_device
from train import make_transform


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cf2_wrd.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--output", default="figures/tsne_reproduced.png")
    ap.add_argument("--max-samples", type=int, default=4000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    ckpt = load_checkpoint(args.checkpoint, "cpu")
    model = build_model(
        cfg,
        selected_local_indices=ckpt["selected_local_indices"],
        pretrained_override=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    ds = WheatRustDataset(
        cfg["data"][f"{args.split}_csv"],
        root=cfg["data"].get("root", "."),
        transform=make_transform(cfg, train=False),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg["data"].get("num_workers", 4))

    feats, labels = [], []
    for batch in loader:
        gx = batch["global_image"].to(device)
        lx = batch["local_image"].to(device)
        _, f = model(gx, lx, return_features=True)
        feats.append(f["fused_feature"].cpu().numpy())
        labels.extend(batch["label"].numpy().tolist())
        if sum(len(x) for x in feats) >= args.max_samples:
            break

    x = np.concatenate(feats, axis=0)[: args.max_samples]
    y = np.asarray(labels)[: len(x)]
    z = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(x)

    plt.figure(figsize=(8, 6))
    for cls, name in [(0, "Healthy"), (1, "Rust")]:
        m = y == cls
        plt.scatter(z[m, 0], z[m, 1], s=10, alpha=0.7, label=name)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
