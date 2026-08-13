#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
import time
import torch

from cf2wrd.model import build_model
from cf2wrd.utils import load_checkpoint, load_config, resolve_device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cf2_wrd.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=500)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, "cpu")
    model = build_model(
        cfg,
        selected_local_indices=ckpt["selected_local_indices"],
        pretrained_override=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    size = cfg["data"].get("input_size", 224)
    gx = torch.randn(1, 3, size, size, device=device)
    lx = torch.randn(1, 3, size, size, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(gx, lx)
        synchronize(device)

        t0 = time.perf_counter()
        for _ in range(args.runs):
            _ = model(gx, lx)
        synchronize(device)
        elapsed = time.perf_counter() - t0

    print(f"Device: {device}")
    print(f"Warm-up runs: {args.warmup}")
    print(f"Timed runs: {args.runs}")
    print(f"Mean latency: {1000*elapsed/args.runs:.3f} ms")
    print(f"FPS: {args.runs/elapsed:.3f}")
    print("Note: model-only timing; image capture and Lab/K-means preprocessing are excluded.")


if __name__ == "__main__":
    main()
