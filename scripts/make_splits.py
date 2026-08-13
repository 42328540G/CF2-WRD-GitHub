#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--output-dir", default="data/splits")
    ap.add_argument("--group-col", default="group_id")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.metadata)
    if args.group_col not in df.columns:
        raise ValueError(
            f"{args.group_col!r} is required. The paper specifies splitting by plant "
            "or acquisition batch; row-wise random splitting is intentionally not used."
        )

    groups = df[args.group_col].astype(str)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=args.seed)
    train_idx, rem_idx = next(gss1.split(df, groups=groups))
    train = df.iloc[train_idx].copy()
    rem = df.iloc[rem_idx].copy()

    rem_groups = rem[args.group_col].astype(str)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=args.seed)
    val_idx, test_idx = next(gss2.split(rem, groups=rem_groups))
    val = rem.iloc[val_idx].copy()
    test = rem.iloc[test_idx].copy()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print(
        "Note: GroupShuffleSplit targets an 80/10/10 group-wise split; exact image "
        "counts can differ from the manuscript unless the original split identifiers are used."
    )


if __name__ == "__main__":
    main()
