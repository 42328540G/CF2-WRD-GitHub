# Final CF2-WRD Method Specification

This document records the method definition used by the v1.0 public reference implementation and the final manuscript text.

## 1. Task

Binary image-level classification:

- `0`: healthy wheat
- `1`: wheat stripe rust

Growth-stage labels are metadata used for subgroup analysis, not output classes.

## 2. Input and preprocessing

- Input size: 224 × 224 RGB.
- Training augmentation: rotation ±15°, horizontal flip, Gaussian blur, brightness perturbation.
- Two synchronized branch inputs:
  - global branch: RGB view;
  - local branch: lesion-enhanced RGB view.
- Local lesion enhancement:
  - RGB → Lab;
  - K-means, K=3;
  - select yellow/orange-yellow candidate using a*/b* color and area information;
  - 3×3 erosion and dilation;
  - use the resulting mask to emphasize lesion-related pixels.

The manuscript does not specify all numerical cluster-score constants, flip/blur probabilities, brightness magnitude, or AdamW weight decay. Those values are explicit repository configuration defaults.

## 3. Global branch

- Backbone: TorchVision MobileNetV2.
- ImageNet pretrained.
- Freeze first 3 feature blocks.
- Final convolutional feature map → global average pooling.
- Linear projection: 1280 → 512.
- Output: `F_G ∈ R^512`.

## 4. Local branch

- Backbone: TorchVision EfficientNet-B0.
- ImageNet pretrained.
- Freeze first 5 feature blocks.
- Final convolutional feature map → global average pooling.
- Linear projection: 1280 → 1024.
- Output: `F_L ∈ R^1024`.

No intermediate multi-scale maps are fused in the final implementation.

## 5. Pearson complementary feature selection

Fit Pearson correlations using the training split only.

For each global dimension i and local dimension j, compute `rho_ij`. Define:

`m_j = max_i |rho_ij|`.

Policy:

- keep all 512 global dimensions;
- activate local dimension j iff `m_j <= tau`;
- use `tau = 0.7`;
- apply the resulting local activity-index mask unchanged to validation and test data;
- no minimum-number-of-local-dimensions fallback.

The reference training script fits this selector on deterministic, non-augmented training views before joint optimization and then freezes the mask. The mask controls active forward-computation dimensions but does not structurally remove parameters.

## 6. ACFFM

- Align full global descriptor to 512-D.
- Keep a full 1024→512 local alignment parameterization and compute the local projection using the active local inputs plus the corresponding weight columns.
- Learn one scalar branch score from each aligned descriptor.
- Apply Softmax over the two scores to obtain `alpha`, `beta`, with `alpha + beta = 1`.
- Fuse by weighted summation:
  `F_fused = alpha * F'_G + beta * F'_L`.

## 7. Classification

- BatchNorm1d(512)
- Dropout p=0.25
- Linear 512 → 2 logits
- Softmax only for reporting probabilities

## 8. Training settings reported in the manuscript

- AdamW
- initial learning rate: 1e-4
- decay ×0.9 every 10 epochs
- batch size: 64
- epochs: 60
- cross-entropy loss
- label smoothing: 0.1
- random seed: 42

## 9. Quantization/deployment settings reported in the manuscript

- INT8 calibration set: 1000 training images
  - 500 seedling
  - 500 adult
- benchmark batch size: 1
- 50 warm-up runs
- 500 timed runs
- FPS excludes image acquisition time

## 10. Parameter count and numerical-reproduction boundary

The finalized architecture has a fixed total of **8,989,568 parameters (8.99 M, rounded)**. This count is invariant to the number of active local dimensions because Pearson selection is implemented as an activity mask/sparse projection over a full 1024→512 local alignment weight matrix. INT8 quantization changes numerical precision, not parameter count.

The public repository does not include the private training dataset or manuscript checkpoint, so manuscript accuracy/F1/FPS results are reported values rather than independently regenerated outputs of this release.