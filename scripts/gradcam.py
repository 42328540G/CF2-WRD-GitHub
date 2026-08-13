#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt

from cf2wrd.model import build_model
from cf2wrd.utils import load_checkpoint, load_config, resolve_device
from train import make_transform


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward)
        self.h2 = target_layer.register_full_backward_hook(self._backward)

    def _forward(self, module, inp, out):
        self.activations = out

    def _backward(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, gx, lx, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(gx, lx)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[:, class_idx].sum().backward()

        a = self.activations
        g = self.gradients
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * a).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=gx.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam, class_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cf2_wrd.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--branch", choices=["mobile", "efficient"], default="efficient")
    ap.add_argument("--output", default="figures/gradcam_example.png")
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

    original = Image.open(args.image).convert("RGB")
    views = make_transform(cfg, train=False)(original)
    gx = views["global_image"].unsqueeze(0).to(device)
    lx = views["local_image"].unsqueeze(0).to(device)

    target = (
        model.encoder.mobile.features[-1]
        if args.branch == "mobile"
        else model.encoder.efficient.features[-1]
    )
    cammer = GradCAM(model, target)
    cam, cls = cammer(gx, lx)
    cammer.close()

    base = np.asarray(original.resize((cfg["data"]["input_size"], cfg["data"]["input_size"])))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.55 * base + 0.45 * heat, 0, 255).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"CF2-WRD Grad-CAM ({args.branch}, class={cls})")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
