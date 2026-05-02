import tensorflow as tf
import numpy as np
import argparse
import os

def quantize_v4l_rocks_int8(model_path, calib_path, output_path):
    """
    Quantization script for MobileNetV4-Large (Rocks) - INT8.
    Standard NHWC alignment for NPU optimization.
    """
    print(f"Loading calibration data: {calib_path}")
    # Load NCHW [100, 3, 244, 244] and transpose to NHWC [100, 244, 244, 3]
    calib_data = np.load(calib_path).astype(np.float32).transpose(0, 2, 3, 1)
    
    def representative_dataset():
        for i in range(len(calib_data)):
            yield [calib_data[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.representative_dataset = representative_dataset
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Force Full Integer I/O
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    print("Converting MobileNetV4-Large Rocks to INT8 TFLite...")
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Successfully saved INT8 model to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lithotheque V4L Rocks INT8 Quantizer")
    parser.add_argument("--model", type=str, required=True, help="Path to FP32 SavedModel")
    parser.add_argument("--calib", type=str, default="calib_roches_244.npy", help="Path to calibration .npy")
    parser.add_argument("--out", type=str, default="roches_v4l_int8.tflite", help="Output path")
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        quantize_v4l_rocks_int8(args.model, args.calib, args.out)
    else:
        print(f"Error: {args.model} not found.")
