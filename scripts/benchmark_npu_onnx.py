import os
import glob
import json
import time
import re
import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image

def clean(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())

def resolve_label(label_raw: str, c104_map: dict, synonyms: dict):
    ctl = clean(label_raw)
    if ctl in c104_map:
        return c104_map[ctl]
    if ctl in synonyms and synonyms[ctl] in c104_map:
        return c104_map[synonyms[ctl]]
    for k, v in c104_map.items():
        if ctl in k or k in ctl:
            return v
    return None

def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX models on Snapdragon Hexagon NPU via QNN Execution Provider.")
    parser.add_argument("--classes", type=str, default="classes_104_en.json", help="Path to classes JSON list.")
    parser.add_argument("--test_dir", type=str, default="test_images", help="Directory containing test images.")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory containing .onnx models.")
    parser.add_argument("--backend_path", type=str, default="QnnHtp.dll", help="Path or library name for QNN HTP backend.")
    parser.add_argument("--output_json", type=str, default="benchmark_results_npu.json", help="File path to store output metrics.")
    args = parser.parse_args()

    if not os.path.exists(args.classes):
        print(f"[!] Warning: Classes file '{args.classes}' not found. Please provide valid path.")
        return

    with open(args.classes, "r", encoding="utf-8") as f:
        classes = json.load(f)

    c104_map = {clean(x): idx for idx, x in enumerate(classes)}

    synonyms = {
        'coquine': clean('Coquina (Shell limestone)'),
        'dolomiticlimestone': clean('Dolomite limestone'),
        'shalemudstone': clean('Shales'),
        'stibnite': clean('Stibine'),
        'breccia': clean('Breach'),
        'chert': clean('Chert (Silexite)'),
        'hornfels': clean('Hornfels (Cornea)'),
        'novaculite': clean('Novaculitis'),
        'pillow': clean('Pillow lava (Pillow)'),
        'silver': clean('Money'),
        'tufa': clean('Tuff (Liestone)'),
        'tuff': clean('Tuff (Volcanic)'),
        'schists': clean('Schist (Limonite)')
    }

    img_files = sorted(glob.glob(os.path.join(args.test_dir, "*.jpg"))) + sorted(glob.glob(os.path.join(args.test_dir, "*.png")))
    print(f"Loaded {len(img_files)} test images for evaluation.")

    qnn_options = {
        "backend_path": args.backend_path,
        "htp_performance_mode": "burst",
        "htp_graph_finalization_optimization_mode": "3"
    }

    models_to_bench = [
        ("MobileNetV4 INT8", os.path.join(args.models_dir, "rock_model_int8_MobileNetV4.onnx"), "INT8"),
        ("MobileNetV4 W4A8", os.path.join(args.models_dir, "rock_model_w4a8_MobileNetV4.onnx"), "W4A8")
    ]

    results_summary = {}

    for label, model_path, fmt in models_to_bench:
        if not os.path.exists(model_path):
            print(f"[-] Skipping {label}: '{model_path}' not found.")
            continue

        print(f"\n=======================================================")
        print(f"STARTING EVALUATION: {label} on Hexagon NPU")
        print(f"=======================================================")
        
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["QNNExecutionProvider", "CPUExecutionProvider"],
            provider_options=[qnn_options, {}]
        )
        print(f"Active Provider: {session.get_providers()[0]}")
        
        inp_name = session.get_inputs()[0].name
        actual_shape = session.get_inputs()[0].shape
        inp_type = session.get_inputs()[0].type
        out_name = session.get_outputs()[0].name
        print(f"Input Name: {inp_name} | Type: {inp_type} | Shape: {actual_shape}")

        is_nchw = (len(actual_shape) == 4 and actual_shape[1] == 3)
        if is_nchw:
            H, W = actual_shape[2], actual_shape[3]
        else:
            H, W = actual_shape[1], actual_shape[2]

        print(f"Target resolution: {H}x{W} (Layout: {'NCHW' if is_nchw else 'NHWC'})")

        top1_correct = 0
        top3_correct = 0
        total_evaluated = 0
        latencies = []
        
        # Warmup passes
        dummy = np.random.randint(-128, 127, size=[d if isinstance(d, int) and d > 0 else 1 for d in actual_shape], dtype=np.int8)
        for _ in range(5):
            _ = session.run([out_name], {inp_name: dummy})

        for idx, img_path in enumerate(img_files):
            base = os.path.splitext(os.path.basename(img_path))[0]
            parts = base.split("_", 2)
            if len(parts) < 3:
                continue
            true_label_raw = parts[2]
            true_idx = resolve_label(true_label_raw, c104_map, synonyms)
            if true_idx is None:
                continue

            img = Image.open(img_path).convert("RGB").resize((W, H))
            arr = np.array(img, dtype=np.float32)
            norm_arr = (arr - 128.0) / 127.5
            q_inp = np.clip(np.round(norm_arr / 0.008305505849421024) + 127, -128, 127).astype(np.int8)

            if is_nchw:
                q_inp = np.transpose(q_inp, (2, 0, 1))
            q_inp = np.expand_dims(q_inp, axis=0)

            t0 = time.perf_counter()
            res = session.run([out_name], {inp_name: q_inp})[0]
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            latencies.append(lat_ms)

            logits = res[0]
            top_indices = np.argsort(logits)[::-1]
            
            if top_indices[0] == true_idx:
                top1_correct += 1
            if true_idx in top_indices[:3]:
                top3_correct += 1
            total_evaluated += 1

        top1_acc = (top1_correct / total_evaluated) * 100.0 if total_evaluated > 0 else 0.0
        top3_acc = (top3_correct / total_evaluated) * 100.0 if total_evaluated > 0 else 0.0
        mean_lat = np.mean(latencies) if latencies else 0.0
        min_lat = np.min(latencies) if latencies else 0.0
        max_lat = np.max(latencies) if latencies else 0.0
        p95_lat = np.percentile(latencies, 95) if latencies else 0.0

        results_summary[label] = {
            "total": total_evaluated,
            "top1": top1_acc,
            "top3": top3_acc,
            "mean_ms": mean_lat,
            "min_ms": min_lat,
            "max_ms": max_lat,
            "p95_ms": p95_lat,
            "fps": 1000.0 / mean_lat if mean_lat > 0 else 0.0
        }

        print(f"Results for {label}:")
        print(f"  Evaluated Images : {total_evaluated}")
        print(f"  Top-1 Accuracy   : {top1_acc:.2f}% ({top1_correct}/{total_evaluated})")
        print(f"  Top-3 Accuracy   : {top3_acc:.2f}% ({top3_correct}/{total_evaluated})")
        print(f"  Mean Latency     : {mean_lat:.2f} ms (~{1000.0/mean_lat:.1f} FPS)")

    print("\n" + "=" * 60)
    print("FINAL BENCHMARK COMPARISON:")
    print("=" * 60)
    for k, v in results_summary.items():
        print(f"{k}: Top-1={v['top1']:.2f}% | Top-3={v['top3']:.2f}% | Latency={v['mean_ms']:.2f} ms ({v['fps']:.1f} FPS)")
    print("=" * 60)

    with open(args.output_json, "w", encoding="utf-8") as f_out:
        json.dump(results_summary, f_out, indent=2)
    print(f"[✓] Benchmark metrics saved to {args.output_json}")

if __name__ == "__main__":
    main()
