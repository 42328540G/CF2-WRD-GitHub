#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from cf2wrd.data import WheatRustDataset
from cf2wrd.model import build_model
from cf2wrd.utils import classification_metrics, load_checkpoint, load_config, resolve_device
from train import make_transform


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cf2_wrd.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
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

    csv_path = cfg["data"][f"{args.split}_csv"]
    ds = WheatRustDataset(
        csv_path,
        root=cfg["data"].get("root", "."),
        transform=make_transform(cfg, train=False),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["training"].get("batch_size", 64),
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
    )

    y_true, y_pred = [], []
    stage = defaultdict(lambda: [[], []])

    for batch in loader:
        gx = batch["global_image"].to(device)
        lx = batch["local_image"].to(device)
        y = batch["label"].to(device)
        pred = model(gx, lx).argmax(1)

        yt, yp = y.cpu().tolist(), pred.cpu().tolist()
        y_true.extend(yt)
        y_pred.extend(yp)

        if "growth_stage" in batch:
            for s, a, b in zip(batch["growth_stage"], yt, yp):
                stage[str(s)][0].append(a)
                stage[str(s)][1].append(b)

    print("Overall:", classification_metrics(y_true, y_pred))
    for s, (a, b) in stage.items():
        print(f"{s}:", classification_metrics(a, b))


if __name__ == "__main__":
    main()
