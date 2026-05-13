# METHODOLOGY: W4A8 Quantization Pipeline

This document details the technical steps, tools, and configurations used to generate the 12 quantized models optimized for the **Snapdragon X Elite** NPU.

---

## 1. Architectural Overview
The target models are:
- **MobileNet V4-S (Scale)**: Lightweight model optimized for high-speed metrology.
- **MobileNet V4-L (Rocks)**: Large model for higher geological precision.
- **MobileNet V5-300M (Rocks)**: Latest MobileNet architecture utilizing GELU activations and optimized bottleneck layers (analogous to the Gemma 3 Nano vision encoder).

---

## 2. Quantization Formats
Two specific formats were produced for each architecture to support the tiered deployment strategy:
1.  **INT8 Full Integer (A8W8)**: 8-bit Activations, 8-bit Weights. Inputs and Outputs forced to INT8.
2.  **W4A8 Full Integer (A8W4)**: 8-bit Activations, 4-bit Weights. Inputs and Outputs forced to INT8.

---

## 3. Mathematical Foundations

The quantization uses **linear symmetric/asymmetric** mapping. The relationship between a floating-point value $q_f$ and its quantized integer value $q_i$ is:

$$q_f = \text{scale} \times (q_i - \text{zero\_point})$$

Where:
- **Scale**: Floating-point scaling factor.
- **Zero Point**: Integer offset (typically 0 or -128 for INT8).

For **W4A8** models:
- **Weights (W4)**: Weights are grouped and quantized block-wise to minimize precision loss in deep convolutional layers.
- **Activations (A8)**: Activations use static calibration (observed min-max) to minimize saturation of outlier values.

### MobileNet V4 (S & L) Pipeline:
- **Tool**: `ai-edge-quantizer` (Google).
- **Process**: 
  - Conversion from `SavedModel` to FP32 TFLite with Signatures.
  - Static calibration on 100 representative geological samples.
  - Export with `Full Integer` flag to ensure zero floating-point operators.
- **I/O Optimization**: Used the `ai-edge-quantizer` API to specify `input_type=int8` and `output_type=int8`.

### MobileNet V5 Specifics:
- **Challenge**: Presence of `GELU` (Erf) operations and memory-intensive tracking of 300M parameters.
- **NPU Solution**: Implemented **Surgical Graph Optimization** to replace `Erf` with a polynomial approximation ($x \times \sigma(1.702 \times x)$). 
- **Graph Freezing**: To ensure the model is fully compatible with LiteRT without triggering `_DictWrapper` or `untracked resource` errors during conversion, all 294M variables were **frozen into constants** directly in the graph definition.
- **4-bit Quantization**: Executed via the `ai-edge-quantizer` compression API, allowing weight reduction to 4-bits while maintaining 8-bit activations for NPU compatibility. This results in a "No-Flex" artifact where 100% of the operators are natively supported by the NPU.

---

## 4. ONNX Quantization Details (ONNX Runtime)

### General Pipeline:
`FP32 ONNX` -> `ONNX Runtime Quantization` -> `Graph Stripping`.

### Calibration Data:
- **Scale**: `calib_echelle.npy` (100 samples, NCHW [1, 3, 1024, 1024]).
- **Rocks V4**: `calib_roches_244.npy` (100 samples, NCHW [1, 3, 244, 244]).
- **Rocks V5**: `calib_roches_256.npy` (100 samples, NCHW [1, 3, 256, 256], transposed to NHWC).

### Forcing Full Integer I/O:
Standard ONNX tools insert `QuantizeLinear` (input) and `DequantizeLinear` (output) nodes that expect floats. To eliminate CPU-bound type casting:
- A custom post-processing script stripped these boundary nodes.
- Graph metadata was manually adjusted to declare `INT8` as the native I/O type.

### Calibration Dataset Preparation
To ensure bit-accurate min-max profiling, we prepared a calibration dataset of 100 representative samples. The following script was used to generate the `.npy` tensors:

```python
import os, glob
import numpy as np
from PIL import Image

def prepare_calib(image_dir, output_path, size=(256, 256), num_samples=100):
    all_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_files.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
    
    np.random.seed(42)
    np.random.shuffle(all_files)
    selected = all_files[:num_samples]
    
    images = []
    for f in selected:
        img = Image.open(f).convert('RGB').resize(size, Image.Resampling.LANCZOS)
        img_data = np.array(img).astype(np.float32) / 255.0
        # NCHW format
        img_data = np.transpose(img_data, (2, 0, 1)) 
        images.append(img_data)
            
    np.save(output_path, np.stack(images))

# Usage for MobileNetV5 (256x256)
# prepare_calib("path/to/images", "calib_roches_256.npy", size=(256, 256))
```

---

## 5. Calibration & Preprocessing

### Data Selection:
We selected **100 images** per model from the validation datasets.
- **Representativeness**: Images cover various lighting conditions and rock textures to ensure the `min/max` calculation during calibration is robust (minimizing KL Divergence).
- **Source Format**: `.npy` (Numpy) files containing normalized tensors.

| Parameter | Scale V4-S | Rocks V4-L | Rocks V5 |
| :--- | :--- | :--- | :--- |
| **Resolution** | **1024 x 1024** | 244 x 244 | 256 x 256 |
| **ONNX Format** | NCHW (1,3,1024,1024) | NCHW (1,3,244,244) | **NHWC** (1,256,256,3) |
| **Normalization** | [0, 1] | [0, 1] | [0, 1] |

---

## 6. Edge Hardware Optimization & Portability

While benchmarks were conducted on the Snapdragon X Elite, the optimization strategy was designed for **maximum cross-platform portability**:
- **W4A8 Format Selection**: The **W4A8** format was chosen over more specialized formats (like W4A16) to ensure the widest possible compatibility across various Edge AI runtimes (LiteRT, ONNX Runtime) and hardware accelerators (NPUs from multiple vendors, modern GPUs). This ensures a consistent performance baseline across the fragmented Android and Windows ARM64 ecosystems.
- **Full Integer I/O**: Without this, the CPU would perform a "Casting" overhead for every frame. By forcing INT8, the data flow remains entirely in the integer domain from the sensor to the final prediction, significantly reducing thermal throttling risks on mobile devices.

---

## 7. Troubleshooting

### A. Erf / GELU (MobileNet V5)
`Erf` is not natively supported by most 8-bit accelerators.
- **Solution**: Removed `SELECT_TF_OPS` in favor of a native subgraph approximation. This ensures that the Qualcomm Hexagon NPU can execute the entire model without falling back to the CPU for activation kernels.

### B. Mismatch in Dimensions
V4 and V5 models use different conventions (NCHW vs NHWC).
- **Solution**: Implemented a dynamic `transpose=(0, 2, 3, 1)` in the `CalibrationDataReader` to adapt on-the-fly without modifying source `.npy` files.

---

## 8. Technical Implementation (Python API)

### A. TFLite Quantization (LiteRT)
Using `ai-edge-quantizer`:

#### 1. Rock Classification (MobileNetV5 - Gemma 3 Nano)
```python
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def export_to_noflex_tflite(keras_model, output_path):
    # 1. Capture model in a concrete function
    @tf.function(input_signature=[tf.TensorSpec([1, 256, 256, 3], tf.float32, name="input_1")])
    def serve(x):
        return {"output_0": keras_model(x, training=False)}

    # 2. Freeze 300M variables into constants (No-Flex requirement)
    frozen_func = convert_variables_to_constants_v2(serve.get_concrete_function())
    
    # 3. Convert via SavedModel with explicit signatures
    temp_sm = "temp_sm"
    tf.saved_model.save(tf.Module(), temp_sm, signatures={"serving_default": frozen_func})
    
    converter = tf.lite.TFLiteConverter.from_saved_model(temp_sm)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
```

#### 2. W4A8 Compression (Weights 4-bit)
Using the `ai_edge_quantizer` API for high-density compression:
```python
compression_config = ai_edge_quantizer.CompressionConfig(
    weight_bits=4,
    activation_bits=8,
    asymmetric=True
)
quantizer.export(
    output_path,
    compression_config=compression_config,
    input_type=ai_edge_quantizer.DType.INT8,
    output_type=ai_edge_quantizer.DType.INT8
)
```

---

### B. Custom Graph Stripping (ONNX Full Integer)
Python logic used to prune dequantization artifacts and force INT8 I/O:

```python
import onnx

def force_full_integer_io(model_path):
    m = onnx.load(model_path)
    g = m.graph
    # Input Node Processing
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
    # Output Node Processing
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
```

---

## 9. Final I/O Specifications
To ensure bit-accurate inference on edge accelerators, all models adhere to the following I/O contracts.

| Model | Format | Input Shape | Output Shape | Axis Order |
| :--- | :--- | :--- | :--- | :--- |
| **Scale V4-S** | TFLite | `[1, 1024, 1024, 3]` | `[1, 1]` | NHWC |
| **Scale V4-S** | ONNX | `[1, 3, 1024, 1024]` | `[1, 1]` | NCHW |
| **Rocks V4-L** | TFLite | `[1, 244, 244, 3]` | `[1, 104]` | NHWC |
| **Rocks V4-L** | ONNX | `[1, 3, 244, 244]` | `[1, 104]` | NCHW |
| **Rocks V5** | TFLite | `[1, 256, 3, 256]` | `[1, 104]` | **NHCW** |
| **Rocks V5** | ONNX | `[1, 256, 256, 3]` | `[1, 104]` | NHWC |

### Input Normalization
All inputs are expected as **Signed INT8** tensors in the range `[-128, 127]`.
```python
# Formula for bit-accurate preprocessing (NumPy)
# Maps [0, 1] range to [-128, 127] signed INT8
input_int8 = (pixel_float * 255 - 128).astype(np.int8)
```

---

## 10. Advanced Inference: 21-Pass Multi-Scale Tiling (21-MST)
To prevent the loss of micro-mineralogical information (like small crystals) during downscaling to 256x256, the Lithotheque engine implements a **21-Pass Multi-Scale Tiling** strategy. 

Instead of a single prediction, the model evaluates 21 distinct crops of the high-resolution input:
- **1 Global Pass**: Full resize (20% weight).
- **4 Medium Passes**: 2x2 grid crops (30% total weight).
- **16 Fine Passes**: 4x4 grid crops (50% total weight).

🔗 **[Reference Python Implementation (scripts/litho_reference_engine.py)](scripts/litho_reference_engine.py)**

---

## 11. Confidence Calibration: Super-Selectivity Matrix

To ensure consistent behavior across diverse hardware tiers and quantization formats, the inference engine applies a dynamic **Super-Selectivity Matrix** during post-processing. This matrix adjusts the Softmax temperature (multiplier) based on the model's `outputScale`, compensating for precision loss in highly compressed (Legacy/Balanced) or ultra-precise (Premium) artifacts.

| Tier / Model Profile | Scale Threshold | Multiplier | Rationale |
| :--- | :--- | :---: | :--- |
| **Premium / App Internal** | `< 0.0003` | **1.500f** | High precision, slight sharpening of peaks. |
| **Standard Rocks** | `< 0.0050` | **1.200f** | Balanced calibration for INT8. |
| **Probe INT8** | `< 0.0100` | **1.000f** | Identity (raw model selectivity). |
| **Balanced Rocks** | `< 0.0240` | **0.800f** | Compensates for INT4 quantization noise. |
| **Legacy Rocks** | `< 0.0300` | **0.700f** | Maximum smoothing for entry-level devices. |
| **Premium Scale** | `< 0.0540` | **5.000f** | High selectivity for geometric metrology. |
| **Std/Legacy Scale** | `< 0.0600` | **4.500f** | Metric precision for fallback tiers. |
| **Default / Unknown** | `else` | **4.000f** | Safety net for new/unidentified models. |

---

## 12. Technical Reproduction Scripts

> [!NOTE]
> **Methodological Blueprint & Adaptability**
> The scripts provided in the `scripts/` directory serve as a technical blueprint. The specific resolutions (e.g., 256x256, 1024x1024), axis orderings, and normalization logic used are **arbitrary examples** based on the Lithotheque engine. Researchers are expected to **adapt these scripts** (specifically input dimensions and normalization layers) to match their own specific models and datasets.

For full scientific reproducibility, the following specialized automation scripts are provided in the `scripts/` directory:

### TFLite / LiteRT (ai-edge-quantizer)
- **MobileNetV5 Rocks**: `tflite_v5_rocks_int8.py` / `tflite_v5_rocks_int4.py`
- **MobileNetV4-L Rocks**: `tflite_v4l_rocks_int8.py` / `tflite_v4l_rocks_int4.py`
- **MobileNetV4-S Scale**: `tflite_v4s_scale_int8.py` / `tflite_v4s_scale_int4.py`

### ONNX (onnxruntime-quantization)
- **MobileNetV5 Rocks**: `onnx_v5_rocks_int8.py` / `onnx_v5_rocks_int4.py`
- **MobileNetV4-L Rocks**: `onnx_v4l_rocks_int8.py` / `onnx_v4l_rocks_int4.py`
- **MobileNetV4-S Scale**: `onnx_v4s_scale_int8.py` / `onnx_v4s_scale_int4.py`

---

## 13. Replication Environment

### Software Stack
- **Python**: 3.11.9 (x64 via Prism)
- **Frameworks**: `tensorflow==2.21.0`, `ai-edge-quantizer==0.5`, `onnxruntime-quantization==1.25.0`
- **Compiler**: MSVC 14.38

### Hardware (Generation Workstation)
- **Platform**: Microsoft Surface Laptop 7 (Snapdragon X Elite X1E80100)
- **NPU**: Qualcomm Hexagon (45 TOPS INT8)
- **RAM**: 32 GB LPDDR5x @ 8448 MT/s

---
*This document certifies the technical reproducibility of the 12 models.*
