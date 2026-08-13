# Deployment Notes

The manuscript reports:

- PyTorch training;
- conversion to TensorFlow Lite;
- INT8 calibration using 1,000 representative training images:
  - 500 seedling-stage;
  - 500 adult-stage;
- batch size 1;
- 50 warm-up runs;
- 500 timed inference runs;
- model-only FPS, excluding image acquisition time.

Reported results:

| Model | Params (M) | Samsung SM-9260 | Jetson TX2 |
|---|---:|---:|---:|
| CF2-WRD | 8.99 | 16.9 FPS | 12.8 FPS |
| CF2-WRD INT8 | 8.99 | 23.0 FPS | 19.2 FPS |

Model-file size is not reported because the finalized deployment artifact is not included; exported size should be measured directly from the actual artifact.

## Included utilities

- `scripts/export_onnx.py`: PyTorch → ONNX export.
- `tflite_int8_template.py`: representative-dataset INT8 quantization after a TensorFlow SavedModel is available.
- `scripts/benchmark.py`: 50 warm-up / 500 timed-run model-only benchmark.

## Reproducibility boundary

The supplied manuscript materials do not specify the exact PyTorch→TensorFlow Lite conversion commands used in the reported experiment. This repository therefore labels the TFLite path as a template rather than claiming that a guessed conversion stack reproduces the published mobile binary exactly.

