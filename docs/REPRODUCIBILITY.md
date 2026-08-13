# Reproducibility Checklist

## Directly specified by the final manuscript and implemented here

- [x] Binary healthy/rust classification
- [x] Input size 224 × 224
- [x] MobileNetV2 global branch
- [x] EfficientNet-B0 local branch
- [x] ImageNet initialization
- [x] Freeze first 3 MobileNetV2 feature blocks
- [x] Freeze first 5 EfficientNet-B0 feature blocks
- [x] 512-D global descriptor
- [x] 1024-D local descriptor
- [x] Vector-level fusion only
- [x] Pearson threshold tau = 0.7
- [x] Keep all global dimensions
- [x] Mask/deactivate local dimension j when max_i |rho_ij| > 0.7
- [x] Fit selector using training data only
- [x] Freeze activity mask for validation/test
- [x] ACFFM alignment to 512-D
- [x] Sample-specific Softmax branch weights
- [x] Dropout p = 0.25
- [x] AdamW, lr = 1e-4
- [x] LR decay ×0.9 every 10 epochs
- [x] Batch size 64
- [x] 60 epochs
- [x] Label smoothing 0.1
- [x] Random seed 42
- [x] Dataset split by plant/acquisition batch
- [x] INT8 calibration: 1000 training images
- [x] Benchmark: batch 1, 50 warm-ups, 500 timed runs

## Explicit public-release defaults not numerically specified in the manuscript

These are configurable and must not be presented as recovered original experimental constants:

- [ ] exact K-means lesion-cluster scoring coefficients
- [ ] minimum valid cluster-area ratio
- [ ] local-view background suppression strength
- [ ] horizontal-flip probability
- [ ] Gaussian-blur probability
- [ ] brightness perturbation magnitude
- [ ] AdamW weight decay

## Items required before claiming exact numerical reproduction

- [ ] Release or privately archive the exact train/validation/test split files.
- [ ] Release or privately archive the manuscript checkpoint.
- [x] Verify fixed architecture parameter count: 8,989,568 (8.99 M).
- [ ] Confirm which definition produced the reported average redundancy 0.27.
- [ ] Preserve the exact mobile conversion commands and final INT8 model.
- [ ] Re-run the final code on the manuscript dataset and archive metrics/logs.
- [ ] Add publication DOI/article metadata after acceptance.
- [ ] Choose and add a software license if public reuse is intended.
