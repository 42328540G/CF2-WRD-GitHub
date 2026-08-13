"""TensorFlow Lite INT8 quantization template.

This starts from a TensorFlow SavedModel. The manuscript does not provide the
exact PyTorch -> TensorFlow SavedModel conversion commands, so that conversion
step is intentionally left explicit rather than guessed.

Example:
    python deployment/tflite_int8_template.py \
        --saved-model deployment/saved_model \
        --representative-npz deployment/calibration_inputs.npz \
        --output deployment/cf2_wrd_int8.tflite
"""

import argparse
import numpy as np
import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-model", required=True)
    ap.add_argument("--representative-npz", required=True)
    ap.add_argument("--output", default="deployment/cf2_wrd_int8.tflite")
    args = ap.parse_args()

    data = np.load(args.representative_npz)
    global_images = data["global_image"].astype(np.float32)
    local_images = data["local_image"].astype(np.float32)

    def representative_dataset():
        for g, l in zip(global_images, local_images):
            yield [g[None, ...], l[None, ...]]

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model = converter.convert()

    with open(args.output, "wb") as f:
        f.write(model)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
