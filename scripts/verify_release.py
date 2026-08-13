#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse

from cf2wrd.model import CF2WRD
from cf2wrd.utils import load_config

EXPECTED = {
    "input_size": 224,
    "num_classes": 2,
    "global_dim": 512,
    "local_dim": 1024,
    "fusion_dim": 512,
    "dropout": 0.25,
    "correlation_threshold": 0.7,
    "freeze_mobile_blocks": 3,
    "freeze_efficient_blocks": 5,
    "epochs": 60,
    "batch_size": 64,
    "lr": 1e-4,
    "lr_step_size": 10,
    "lr_gamma": 0.9,
    "label_smoothing": 0.1,
}
EXPECTED_TOTAL_PARAMS = 8_989_568


def nparams(model):
    return sum(p.numel() for p in model.parameters())


def check_config(cfg):
    actual = {
        "input_size": cfg["data"]["input_size"],
        "num_classes": cfg["model"]["num_classes"],
        "global_dim": cfg["model"]["global_dim"],
        "local_dim": cfg["model"]["local_dim"],
        "fusion_dim": cfg["model"]["fusion_dim"],
        "dropout": cfg["model"]["dropout"],
        "correlation_threshold": cfg["model"]["correlation_threshold"],
        "freeze_mobile_blocks": cfg["model"]["freeze_mobile_blocks"],
        "freeze_efficient_blocks": cfg["model"]["freeze_efficient_blocks"],
        "epochs": cfg["training"]["epochs"],
        "batch_size": cfg["training"]["batch_size"],
        "lr": cfg["training"]["lr"],
        "lr_step_size": cfg["training"]["lr_step_size"],
        "lr_gamma": cfg["training"]["lr_gamma"],
        "label_smoothing": cfg["training"]["label_smoothing"],
    }
    failures = []
    for key, expected in EXPECTED.items():
        if actual[key] != expected:
            failures.append(f"{key}: expected {expected!r}, got {actual[key]!r}")
    return failures


def count_for_active(k: int):
    model = CF2WRD(
        pretrained=False,
        num_classes=2,
        global_dim=512,
        local_dim=1024,
        fusion_dim=512,
        dropout=0.25,
        freeze_mobile_blocks=3,
        freeze_efficient_blocks=5,
        selected_local_indices=range(k),
    )
    return nparams(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cf2_wrd.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    failures = check_config(cfg)

    print("CF2-WRD v1.1 release verification")
    print("--------------------------------")
    if failures:
        for item in failures:
            print("CONFIG MISMATCH:", item)
    else:
        print("Manuscript-level configuration: OK")

    print("\nPearson activity policy:")
    print("  keep all global dimensions")
    print("  activate local j iff max_i |rho_ij| <= 0.7")
    print("  use active local inputs + matching full-projection weight columns")
    print("  no minimum-dimension fallback")

    counts=[]
    print("\nFixed architecture parameter-count check:")
    for k in (1, 64, 512, 1024):
        c=count_for_active(k)
        counts.append(c)
        print(f"  active local dims={k:4d}: {c:,} ({c/1e6:.6f} M)")
    if any(c != EXPECTED_TOTAL_PARAMS for c in counts):
        failures.append(f"parameter count is not invariant at {EXPECTED_TOTAL_PARAMS:,}")
    else:
        print(f"  invariant total: {EXPECTED_TOTAL_PARAMS:,} = 8.99 M (rounded)")

    if failures:
        for item in failures:
            print("FAIL:", item)
        raise SystemExit(1)
    print("\nRelease verification: PASS")


if __name__ == "__main__":
    main()
