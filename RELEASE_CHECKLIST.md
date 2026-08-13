# v1.0 Submission Release Checklist

## Completed in this package

- [x] Round6 vector-level method terminology
- [x] Updated Fig. 1 workflow
- [x] Updated Fig. 3 vector-level architecture
- [x] 512-D MobileNetV2 global descriptor
- [x] 1024-D EfficientNet-B0 local descriptor
- [x] Strict final Pearson rule (`m_j <= 0.7`)
- [x] Removed the previous `min_local_dims` fallback
- [x] Training-only selector fit
- [x] Fixed selector mask for validation/test
- [x] 512-D ACFFM alignment and Softmax branch weighting
- [x] Dropout 0.25 and two-class output
- [x] Dataset availability wording updated
- [x] Removed obsolete pre-Round6 manuscript-fix documentation
- [x] Added release verification script and selector tests
- [x] Version metadata updated to 1.0.0

## Must be verified by the authors before claiming exact paper reproduction

- [x] Verify fixed architecture parameter count at 8.99 M
- [ ] Confirm the exact definition used for manuscript `average redundancy = 0.27`
- [ ] Preserve exact original dataset split files
- [ ] Preserve/release the exact manuscript checkpoint if permitted
- [ ] Confirm exact K-means scoring constants used in the experiment
- [ ] Confirm augmentation probabilities/magnitudes used in the experiment
- [ ] Confirm AdamW weight decay used in the experiment
- [ ] Confirm exact PyTorch→TensorFlow Lite conversion path
- [ ] Choose a software license
- [ ] Add DOI/article metadata after publication
