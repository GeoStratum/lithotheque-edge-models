import tensorflow as tf
import numpy as np
import ai_edge_quantizer
import argparse
import os

def quantize_v4s_scale_int4(model_path, calib_path, output_path):
    """
    Quantization script for MobileNetV4-Small (Scale Metrology) - W4A8 (INT4 Weights).
    Optimized for 1024x1024 resolution.
    """
    print(f"Loading calibration data: {calib_path}")
    # Load NCHW and transpose to NHWC
    calib_data = np.load(calib_path).astype(np.float32).transpose(0, 2, 3, 1)
    
    quantizer = ai_edge_quantizer.TFLiteQuantizer(
        model_path=model_path,
        calibration_data=[calib_data]
    )
    
    compression_config = ai_edge_quantizer.CompressionConfig(
        weight_bits=4,
        activation_bits=8,
        asymmetric=True
    )
    
    print("Exporting MobileNetV4-Small Scale to W4A8 (INT4) TFLite...")
    quantizer.export(
        output_path,
        compression_config=compression_config,
        input_type=ai_edge_quantizer.DType.INT8,
        output_type=ai_edge_quantizer.DType.INT8
    )
    print(f"Successfully saved W4A8 model to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lithotheque V4S Scale INT4 Quantizer")
    parser.add_argument("--model", type=str, required=True, help="Path to FP32 SavedModel")
    parser.add_argument("--calib", type=str, default="calib_echelle.npy", help="Path to calibration .npy")
    parser.add_argument("--out", type=str, default="echelle_v4s_int4.tflite", help="Output path")
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        quantize_v4s_scale_int4(args.model, args.calib, args.out)
    else:
        print(f"Error: {args.model} not found.")
