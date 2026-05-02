import tensorflow as tf
import numpy as np
import ai_edge_quantizer
import argparse
import os

def quantize_v5_rocks_int8(model_path, calib_path, output_path):
    """
    Quantization script for MobileNetV5-300M (Rocks) - INT8.
    Optimized for Snapdragon NPU with NHCW axis alignment and Flex Ops support.
    """
    print(f"Loading calibration data: {calib_path}")
    # Load NCHW [100, 3, 256, 256] and transpose to NHCW [100, 256, 3, 256]
    calib_data = np.load(calib_path).astype(np.float32).transpose(0, 2, 1, 3)
    
    def representative_dataset():
        for i in range(len(calib_data)):
            yield [calib_data[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.representative_dataset = representative_dataset
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Force Full Integer I/O
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Enable SELECT_TF_OPS for GELU/Erf compatibility in MobileNetV5
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    print("Converting MobileNetV5 Rocks to INT8 TFLite...")
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Successfully saved INT8 model to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lithotheque V5 Rocks INT8 Quantizer")
    parser.add_argument("--model", type=str, required=True, help="Path to FP32 SavedModel")
    parser.add_argument("--calib", type=str, default="calib_roches_256.npy", help="Path to calibration .npy")
    parser.add_argument("--out", type=str, default="roches_v5_int8.tflite", help="Output path")
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        quantize_v5_rocks_int8(args.model, args.calib, args.out)
    else:
        print(f"Error: {args.model} not found.")
