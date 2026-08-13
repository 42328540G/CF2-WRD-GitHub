#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
from PIL import Image
import torch

from cf2wrd.model import build_model
from cf2wrd.utils import load_checkpoint, load_config, resolve_device
from train import make_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cf2_wrd.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
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

    image = Image.open(args.image).convert("RGB")
    views = make_transform(cfg, train=False)(image)
    gx = views["global_image"].unsqueeze(0).to(device)
    lx = views["local_image"].unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.softmax(model(gx, lx), dim=1)[0].cpu()
    names = ["Healthy", "Stripe rust"]
    pred = int(prob.argmax())
    print(f"Prediction: {names[pred]}")
    print(f"Healthy probability: {prob[0].item():.6f}")
    print(f"Rust probability:    {prob[1].item():.6f}")


if __name__ == "__main__":
    main()
