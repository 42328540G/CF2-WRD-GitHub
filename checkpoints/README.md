# Checkpoints

The public repository does not include the private manuscript training checkpoint.

The training script writes checkpoints containing:

- model state;
- selected local Pearson-feature indices;
- selector correlation summaries;
- configuration;
- validation metrics.

Suggested names:

```text
cf2_wrd_best.pt
cf2_wrd_fp32.pt
cf2_wrd_int8.tflite
```

A manuscript checkpoint should only be published if the authors have verified that it corresponds to the submitted final architecture and are permitted to release it.
