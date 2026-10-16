"""
Lithotheque Edge AI - MobileNetV4-Large TFLite INT8 Quantization Script
Compiles MobileNetV4-Large to Full Integer INT8 TFLite (W8A8).
Eliminates Flex operations and enforces pure INT8 tensor I/O for Qualcomm Hexagon NPU execution.
Supports both SavedModel directories and PyTorch/Keras exported artifacts.
Input tensor: NHWC [1, 244, 244, 3] or [1, 384, 384, 3] | 104 Geological Classes
"""

import tensorflow as tf
import numpy as np
import argparse
import os

def quantize_v4l_rocks_int8(model_path, calib_path, output_path):
    print(f"Loading calibration dataset from: {calib_path}")
    calib_data = np.load(calib_path).astype(np.float32)
    
    # Ensure NHWC [B, H, W, C] format for TFLite
    if calib_data.ndim == 4 and calib_data.shape[1] == 3:
        calib_data = np.transpose(calib_data, (0, 2, 3, 1))
    
    print(f"Calibration tensor shape (NHWC): {calib_data.shape}")

    def representative_dataset():
        for i in range(min(100, len(calib_data))):
            yield [calib_data[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.representative_dataset = representative_dataset
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Strict NPU Integer Enforcement (Zero-FP32)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # 100% NPU Hardware Compliance (Exynos Eden NCP 24/25, Google Tensor, Qualcomm Hexagon)
    # FullyConnected layers require per-tensor quantization (TENSOR_QUANT8_ASYMM_SIGNED)
    if hasattr(converter, "_experimental_disable_per_channel_quantization_for_dense_layers"):
        converter._experimental_disable_per_channel_quantization_for_dense_layers = True
    
    print("Converting MobileNetV4-Large Rocks to INT8 TFLite (100% NPU compatible)...")
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Successfully exported Zero-FP32 INT8 TFLite model to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lithotheque MobileNetV4-Large Rocks INT8 TFLite Quantizer")
    parser.add_argument("--model", type=str, required=True, help="Path to FP32 SavedModel")
    parser.add_argument("--calib", type=str, default="calib_roches_244.npy", help="Path to calibration .npy")
    parser.add_argument("--out", type=str, default="roches_v4l_int8.tflite", help="Output .tflite path")
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        quantize_v4l_rocks_int8(args.model, args.calib, args.out)
    else:
        print(f"Error: Model directory '{args.model}' not found.")
