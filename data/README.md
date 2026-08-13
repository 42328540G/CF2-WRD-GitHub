# Dataset Preparation

The wheat stripe rust image files are not included in this repository.

Expected metadata columns:

| Column | Required | Meaning |
|---|---|---|
| `image_path` | yes | image path relative to data root or absolute path |
| `label` | yes | `healthy`/`rust` or `0`/`1` |
| `growth_stage` | recommended | `seedling` or `adult` |
| `disease_stage` | optional | `healthy`, `early`, `late`, `outbreak`, `severe` |
| `device` | optional | acquisition device |
| `group_id` | strongly recommended | plant ID or acquisition-batch ID |

The manuscript reports:

- train: 51,376
- validation: 6,428
- test: 6,428
- total: 64,232

All images from the same plant or acquisition batch should be assigned to only one split.

If the exact original splits are available, place them at:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

If only unsplit metadata are available, `scripts/make_splits.py` creates a group-aware 80/10/10 split. A newly generated split will not necessarily reproduce the exact manuscript image counts.

According to the manuscript, the datasets are available from the corresponding author on reasonable request for academic and non-commercial research purposes.
