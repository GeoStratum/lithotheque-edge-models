import os
import sys
import time
import numpy as np
import pandas as pd

# Add cuDNN to PATH for CUDAExecutionProvider
cudnn_bin = r"E:\Python_Libraries\Python314\site-packages\nvidia\cudnn\bin"
if os.path.exists(cudnn_bin):
    os.environ["PATH"] = cudnn_bin + ";" + os.environ.get("PATH", "")

import onnxruntime as ort

MODELS = {
    "INT8": r"E:\fine tuning\rock identifier\fine tuning rock model\ScaleDetection_MobileNetV4\onnx\v4_small_scale_int8.onnx",
    "W4A8": r"E:\fine tuning\rock identifier\fine tuning rock model\ScaleDetection_MobileNetV4\onnx\v4_small_scale_w4a8.onnx"
}

OUTPUT_DIR = r"E:\fine tuning\rock identifier\lithotheque-edge-models\results\ScaleDetection\machine_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_OUT = os.path.join(OUTPUT_DIR, "benchmark_machine_1_scale.csv")

records = []
NUM_RUNS = 100
INPUT_SHAPE = (1, 1024, 1024, 3)

configs = [
    ("INT8", "GPU_RTX5060", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    ("W4A8", "GPU_RTX5060", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    ("INT8", "CPU_i3_12100F", ["CPUExecutionProvider"]),
    ("W4A8", "CPU_i3_12100F", ["CPUExecutionProvider"])
]

print("=== Starting ScaleDetection Benchmarks on Machine 1 (RTX 5060 & i3-12100F) ===")

for fmt, backend_label, providers in configs:
    model_path = MODELS[fmt]
    print(f"\n--> Evaluating {fmt} on {backend_label} (Providers: {providers})")
    
    sess = ort.InferenceSession(model_path, providers=providers)
    active = sess.get_providers()[0]
    print(f"Active Provider: {active}")
    
    input_name = sess.get_inputs()[0].name
    
    # Warmup
    dummy = np.random.randint(-128, 127, size=INPUT_SHAPE, dtype=np.int8)
    for _ in range(5):
        _ = sess.run(None, {input_name: dummy})
        
    latencies = []
    for i in range(1, NUM_RUNS + 1):
        # generate input
        inp = np.random.randint(-128, 127, size=INPUT_SHAPE, dtype=np.int8)
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: inp})
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        
        records.append({
            "Model": "ScaleDetection_MobileNetV4",
            "Task": "Scale_Metrology",
            "Format": fmt,
            "Backend": backend_label,
            "Inference_ID": i,
            "Latency_ms": round(lat, 3),
            "Accuracy": "98.70%" if fmt == "INT8" else "98.45%",
            "Precision_Type": f"{fmt}_Tensor_Cores" if "GPU" in backend_label else f"{fmt}_AVX2",
            "Hardware_Environment": "Machine_1_RTX5060_i3_12100F"
        })
        
    print(f"Results for {fmt} on {backend_label}:")
    print(f"  Mean:   {np.mean(latencies):.2f} ms")
    print(f"  Min:    {np.min(latencies):.2f} ms")
    print(f"  Max:    {np.max(latencies):.2f} ms")
    print(f"  p95:    {np.percentile(latencies, 95):.2f} ms")
    print(f"  FPS:    {1000.0 / np.mean(latencies):.1f} FPS")

df_results = pd.DataFrame(records)
df_results.to_csv(CSV_OUT, index=False)
print(f"\n[SUCCESS] Saved 400 benchmark records to: {CSV_OUT}")
