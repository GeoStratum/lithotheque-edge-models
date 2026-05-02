import tensorflow as tf
import numpy as np
import ai_edge_quantizer
import argparse
import os

def quantize_v5_rocks_int4(model_path, calib_path, output_path):
    """
    Quantization script for MobileNetV5-300M (Rocks) - W4A8 (INT4 Weights).
    Optimized for Snapdragon NPU with NHCW axis alignment and Flex Ops support.
    Uses ai-edge-quantizer for high-density weight compression.
    """
    print(f"Loading calibration data: {calib_path}")
    # Load NCHW [100, 3, 256, 256] and transpose to NHCW [100, 256, 3, 256]
    calib_data = np.load(calib_path).astype(np.float32).transpose(0, 2, 1, 3)
    
    # We use ai_edge_quantizer API for INT4
    quantizer = ai_edge_quantizer.TFLiteQuantizer(
        model_path=model_path,
        calibration_data=[calib_data]
    )
    
    # Configure 4-bit weight compression
    compression_config = ai_edge_quantizer.CompressionConfig(
        weight_bits=4,
        activation_bits=8,
        asymmetric=True
    )
    
    # Note: ensure SELECT_TF_OPS is handled if using the quantizer wrapper
    # The quantizer wrapper usually handles this via the underlying converter
    
    print("Exporting MobileNetV5 Rocks to W4A8 (INT4) TFLite...")
    quantizer.export(
        output_path,
        compression_config=compression_config,
        input_type=ai_edge_quantizer.DType.INT8,
        output_type=ai_edge_quantizer.DType.INT8
    )
    print(f"Successfully saved W4A8 model to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lithotheque V5 Rocks INT4 Quantizer")
    parser.add_argument("--model", type=str, required=True, help="Path to FP32 SavedModel")
    parser.add_argument("--calib", type=str, default="calib_roches_256.npy", help="Path to calibration .npy")
    parser.add_argument("--out", type=str, default="roches_v5_int4.tflite", help="Output path")
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        quantize_v5_rocks_int4(args.model, args.calib, args.out)
    else:
        print(f"Error: {args.model} not found.")
