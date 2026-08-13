# Parameter Count Definition

## Final Round 7 architecture

The finalized CF2-WRD vector-level implementation contains **8,989,568 stored parameters**, reported as **8.99 M** after rounding.

The count includes:

- MobileNetV2 feature extractor and 1280→512 global projection;
- EfficientNet-B0 feature extractor and 1280→1024 local projection;
- ACFFM with a fixed 512→512 global alignment and fixed 1024→512 local alignment;
- branch-scoring layers;
- BatchNorm and the 512→2 classifier.

## Why Pearson selection does not change the count

Pearson correlation determines a fixed set of **active local dimensions** from the training split. During forward computation, the model selects those local inputs and the corresponding columns of the full 1024→512 local-alignment weight matrix. The full weight matrix remains part of the stored model. Therefore, changing the number of active dimensions changes effective local-projection computation but does not change the number of stored parameters.

The same parameter count applies to FP32 and INT8 representations; quantization changes numerical precision, not parameter count.

Run:

```bash
python scripts/verify_release.py --config configs/cf2_wrd.yaml
```

to verify the invariant parameter count.

## Model-file size

Model-file size is deliberately not inferred from the 8.99 M parameter count. Exported size depends on graph format, quantization metadata, operator representation, and other serialization details and should be measured from the actual finalized deployment artifact.
