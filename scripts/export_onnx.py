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

from cf2wrd.model import build_model
from cf2wrd.utils import load_checkpoint, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cf2_wrd.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="deployment/cf2_wrd.onnx")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = load_checkpoint(args.checkpoint, "cpu")
    model = build_model(
        cfg,
        selected_local_indices=ckpt["selected_local_indices"],
        pretrained_override=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    size = cfg["data"].get("input_size", 224)
    gx = torch.randn(1, 3, size, size)
    lx = torch.randn(1, 3, size, size)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (gx, lx),
        str(out),
        input_names=["global_image", "local_image"],
        output_names=["logits"],
        dynamic_axes={
            "global_image": {0: "batch"},
            "local_image": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported: {out}")


if __name__ == "__main__":
    main()
