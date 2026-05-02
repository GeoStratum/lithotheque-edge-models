import onnx
import numpy as np
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader
import argparse
import os

class ScaleV4SDataReader(CalibrationDataReader):
    def __init__(self, calib_path):
        # Load NCHW [100, 3, 1024, 1024]
        self.data = np.load(calib_path).astype(np.float32)
        self.enum_data = iter([{'input': self.data[i:i+1]} for i in range(len(self.data))])

    def get_next(self):
        return next(self.enum_data, None)

def force_full_integer_io(model_path):
    m = onnx.load(model_path)
    g = m.graph
    # Strip boundary nodes
    inp = g.input[0]
    for n in list(g.node):
        if n.op_type == 'QuantizeLinear' and inp.name in n.input:
            out_name = n.output[0]
            g.node.remove(n)
            for c in g.node:
                for i, in_c in enumerate(c.input):
                    if in_c == out_name: c.input[i] = inp.name
            inp.type.tensor_type.elem_type = 3 # INT8
            break
    out = g.output[0]
    for n in list(g.node):
        if n.op_type == 'DequantizeLinear' and out.name in n.output:
            in_name = n.input[0]
            g.node.remove(n)
            for p in g.node:
                for i, out_p in enumerate(p.output):
                    if out_p == in_name: p.output[i] = out.name
            out.type.tensor_type.elem_type = 3 # INT8
            break
    onnx.save(m, model_path)

def quantize_v4s_scale_onnx_int4(model_path, calib_path, output_path):
    print(f"Quantizing {model_path} to ONNX INT4 (W4A8)...")
    dr = ScaleV4SDataReader(calib_path)
    quantize_static(
        model_path,
        output_path,
        dr,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt4,
        extra_options={
            'WeightQuantizeType': QuantType.QInt4,
            'EnableSubgraph': True,
            'ForceQuantizeNoInputCheck': True
        }
    )
    force_full_integer_io(output_path)
    print(f"Successfully saved Full Integer INT4 model to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lithotheque V4S Scale ONNX INT4 Quantizer")
    parser.add_argument("--model", type=str, required=True, help="Path to FP32 ONNX model")
    parser.add_argument("--calib", type=str, default="calib_echelle.npy", help="Path to calibration .npy")
    parser.add_argument("--out", type=str, default="echelle_v4s_int4.onnx", help="Output path")
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        quantize_v4s_scale_onnx_int4(args.model, args.calib, args.out)
    else:
        print(f"Error: {args.model} not found.")
