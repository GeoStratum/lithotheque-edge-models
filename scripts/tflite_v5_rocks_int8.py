import tensorflow as tf
import numpy as np
import os
import argparse
import gc
import shutil
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from ai_edge_quantizer import Quantizer
from ai_edge_quantizer.qtyping import TFLOperationName, QuantGranularity
from ai_edge_litert.interpreter import Interpreter

def quantize_v5_rocks_int8(model_path, calib_path, output_path):
    """
    Advanced quantization script for MobileNetV5-300M (Rocks) - INT8.
    Optimized for NPU with Frozen Graph strategy to eliminate Flex Ops and memory bottlenecks.
    """
    temp_sm = "temp_v5_sm"
    temp_float = "temp_v5_float.tflite"

    # Step 1: Create Frozen Float TFLite
    if not os.path.exists(temp_float):
        print("Loading FP32 model and freezing variables...")
        # Assume model_path is a SavedModel or H5. If it's a directory, it's a SavedModel.
        model = tf.keras.models.load_model(model_path)
        
        @tf.function(input_signature=[tf.TensorSpec([1, 256, 256, 3], tf.float32, name="input_1")])
        def serve(x):
            return {"output_0": model(x, training=False)}

        frozen_func = convert_variables_to_constants_v2(serve.get_concrete_function())
        del model; gc.collect()

        if os.path.exists(temp_sm): shutil.rmtree(temp_sm)
        tf.saved_model.save(tf.Module(), temp_sm, signatures={"serving_default": frozen_func})
        
        converter = tf.lite.TFLiteConverter.from_saved_model(temp_sm)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        float_model = converter.convert()
        with open(temp_float, "wb") as f: f.write(float_model)
        shutil.rmtree(temp_sm)
        print("Float TFLite created.")

    # Step 2: Quantize with ai-edge-quantizer (W8A8)
    print(f"Quantizing to INT8: {output_path}")
    q = Quantizer(temp_float)
    
    # Configure INT8
    q.add_static_config(
        regex=".*",
        operation_name=TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=8,
        weight_num_bits=8,
        weight_granularity=QuantGranularity.CHANNELWISE,
    )

    # Calibration data
    raw = np.load(calib_path).astype(np.float32)
    # Adjust axes if needed (assuming NCHW -> NHWC)
    if raw.shape[1] == 3: raw = raw.transpose(0, 2, 3, 1)
    
    # Identify signature keys
    interp = Interpreter(model_path=temp_float)
    sigs = interp.get_signature_list()
    sig = list(sigs.keys())[0] if sigs else "serving_default"
    inp_k = list(sigs[sig]["inputs"].keys())[0] if sigs else interp.get_input_details()[0]["name"].split(":")[0]
    
    calib_dict = {sig: [{inp_k: raw[i:i+1]} for i in range(min(100, len(raw)))]}
    
    result = q.quantize(q.calibrate(calib_dict))
    
    with open(output_path, "wb") as f:
        f.write(result.quantized_model)
    
    print(f"Successfully saved INT8 model to {output_path}")
    if os.path.exists(temp_float): os.remove(temp_float)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lithotheque V5 Rocks INT8 Quantizer")
    parser.add_argument("--model", type=str, required=True, help="Path to FP32 Model (SavedModel/H5)")
    parser.add_argument("--calib", type=str, default="calib_roches_256.npy", help="Path to calibration .npy")
    parser.add_argument("--out", type=str, default="roches_v5_int8_noflex.tflite", help="Output path")
    args = parser.parse_args()
    
    quantize_v5_rocks_int8(args.model, args.calib, args.out)
