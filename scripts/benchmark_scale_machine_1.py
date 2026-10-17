import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Dynamic path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_ROOT", BASE_DIR / "models"))

# Add cuDNN to PATH for CUDAExecutionProvider if present
cudnn_bin = Path(r"E:\Python_Libraries\Python314\site-packages\nvidia\cudnn\bin")
if cudnn_bin.exists():
    os.environ["PATH"] = str(cudnn_bin) + ";" + os.environ.get("PATH", "")

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not installed. Please install onnxruntime-gpu or onnxruntime.")
    sys.exit(0)

OUTPUT_DIR = BASE_DIR / "results" / "ScaleDetection" / "machine_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = OUTPUT_DIR / "benchmark_machine_1_scale.csv"

def find_scale_model(fmt: str) -> Path:
    candidates = [
        MODELS_DIR / "ScaleDetection_MobileNetV4" / "onnx" / f"v4_small_scale_{fmt.lower()}.onnx",
        Path(r"E:\fine tuning\rock identifier\fine tuning rock model\ScaleDetection_MobileNetV4\onnx") / f"v4_small_scale_{fmt.lower()}.onnx",
        BASE_DIR / "models" / f"v4_small_scale_{fmt.lower()}.onnx"
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

NUM_RUNS = 100
INPUT_SHAPE = (1, 1024, 1024, 3)

configs = [
    ("INT8", "GPU_RTX5060", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    ("W4A8", "GPU_RTX5060", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    ("INT8", "CPU_AVX2", ["CPUExecutionProvider"]),
    ("W4A8", "CPU_AVX2", ["CPUExecutionProvider"])
]

print("=== Starting ScaleDetection Benchmarks on Machine 1 ===")

records = []
for fmt, backend_label, providers in configs:
    model_path = find_scale_model(fmt)
    if not model_path.exists():
        print(f"[!] Model not found, skipping: {model_path}")
        continue

    print(f"\n--> Evaluating {fmt} on {backend_label} (Providers: {providers})")
    sess = ort.InferenceSession(str(model_path), providers=providers)
    active = sess.get_providers()[0]
    print(f"Active Provider: {active}")
    input_name = sess.get_inputs()[0].name

    # Warmup
    dummy = np.random.randint(-128, 127, size=INPUT_SHAPE, dtype=np.int8)
    for _ in range(5):
        _ = sess.run(None, {input_name: dummy})

    latencies = []
    for i in range(1, NUM_RUNS + 1):
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
            "Precision_Type": f"{fmt}_Tensor_Cores" if "GPU" in backend_label else f"{fmt}_AVX2",
            "Hardware_Environment": "Machine_1_Workstation"
        })

    print(f"Results for {fmt} on {backend_label}:")
    print(f"  Mean:   {np.mean(latencies):.2f} ms")
    print(f"  Min:    {np.min(latencies):.2f} ms")
    print(f"  Max:    {np.max(latencies):.2f} ms")
    print(f"  p95:    {np.percentile(latencies, 95):.2f} ms")
    print(f"  FPS:    {1000.0 / np.mean(latencies):.1f} FPS")

if records:
    df_results = pd.DataFrame(records)
    df_results.to_csv(CSV_OUT, index=False)
    print(f"\n[SUCCESS] Saved {len(records)} benchmark records to: {CSV_OUT}")

