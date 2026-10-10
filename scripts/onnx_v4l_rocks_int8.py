"""
Lithotheque Edge AI - MobileNetV4-Large INT8 Quantization Script
Quantizes MobileNetV4-Large for rock recognition to full integer INT8 (W8A8).
Enforces Zero-FP32 I/O constraint for seamless Qualcomm Hexagon NPU offloading.
Input tensor: NCHW [1, 3, 244, 244] or [1, 3, 384, 384] | 104 Geological Classes
"""

import onnx
import numpy as np
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader
import argparse
import os

class RocksV4LDataReader(CalibrationDataReader):
    """
    Feeds calibration samples into ONNX static quantizer.
    Expects NCHW shape (e.g. [1, 3, 244, 244] or [1, 3, 384, 384]).
    """
    def __init__(self, calib_path, input_name=None):
        self.data = np.load(calib_path).astype(np.float32)
        # Ensure NCHW format: if channels are last (NHWC), transpose to NCHW
        if self.data.ndim == 4 and self.data.shape[-1] == 3:
            self.data = np.transpose(self.data, (0, 3, 1, 2))
        self.input_name = input_name or 'input'
        self.enum_data = iter([{self.input_name: self.data[i:i+1]} for i in range(len(self.data))])

    def get_next(self):
        return next(self.enum_data, None)

def force_full_integer_io(model_path):
    """
    Strips top-level QuantizeLinear and bottom-level DequantizeLinear wrappers
    to convert the ONNX model into pure Full Integer I/O (pure INT8 graph).
    """
    m = onnx.load(model_path)
    g = m.graph
    
    # Strip boundary input QuantizeLinear
    inp = g.input[0]
    for n in list(g.node):
        if n.op_type == 'QuantizeLinear' and inp.name in n.input:
            out_name = n.output[0]
            g.node.remove(n)
            for c in g.node:
                for i, in_c in enumerate(c.input):
                    if in_c == out_name:
                        c.input[i] = inp.name
            inp.type.tensor_type.elem_type = onnx.TensorProto.INT8
            break

    # Strip boundary output DequantizeLinear
    out = g.output[0]
    for n in list(g.node):
        if n.op_type == 'DequantizeLinear' and out.name in n.output:
            in_name = n.input[0]
            g.node.remove(n)
            for p in g.node:
                for i, out_p in enumerate(p.output):
                    if out_p == in_name:
                        p.output[i] = out.name
            out.type.tensor_type.elem_type = onnx.TensorProto.INT8
            break

    onnx.save(m, model_path)

def quantize_v4l_rocks_onnx_int8(model_path, calib_path, output_path, strip_io=True):
    print(f"Quantizing {model_path} to ONNX INT8 (QDQ / Full-Int)...")
    
    # Inspect model input tensor name
    m_check = onnx.load(model_path)
    inp_name = m_check.graph.input[0].name
    
    dr = RocksV4LDataReader(calib_path, input_name=inp_name)
    quantize_static(
        model_path,
        output_path,
        dr,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8
    )
    
    if strip_io:
        force_full_integer_io(output_path)
        print(f"Zero-FP32 I/O enforced. Saved Full Integer INT8 model to: {output_path}")
    else:
        print(f"Standard QDQ INT8 model saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lithotheque MobileNetV4-Large Rocks ONNX INT8 Quantizer")
    parser.add_argument("--model", type=str, required=True, help="Path to FP32 ONNX model")
    parser.add_argument("--calib", type=str, default="calib_roches_244.npy", help="Path to calibration .npy")
    parser.add_argument("--out", type=str, default="roches_v4l_int8.onnx", help="Output .onnx path")
    parser.add_argument("--preserve-io", action="store_true", help="Preserve float32 IO wrappers (Standard QDQ)")
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        quantize_v4l_rocks_onnx_int8(args.model, args.calib, args.out, strip_io=not args.preserve_io)
    else:
        print(f"Error: Model file '{args.model}' not found.")
