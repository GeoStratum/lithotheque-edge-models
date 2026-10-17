import os
import glob
import json
import time
import re
import subprocess
import argparse
import numpy as np
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
    parser = argparse.ArgumentParser(description="Run on-device/desktop NPU context binary benchmark via Qualcomm QAIRT SDK.")
    parser.add_argument("--qairt_bin", type=str, default="qnn-net-run.exe", help="Path to qnn-net-run executable.")
    parser.add_argument("--backend", type=str, default="QnnHtp.dll", help="Path to QnnHtp backend library.")
    parser.add_argument("--classes", type=str, default="classes_104_en.json", help="Path to classes JSON.")
    parser.add_argument("--test_dir", type=str, default="test_images", help="Directory of evaluation images.")
    parser.add_argument("--work_dir", type=str, default="qairt_eval_workspace", help="Temporary workspace for raw inputs and outputs.")
    parser.add_argument("--bin_int8", type=str, default="", help="Path to INT8 .bin context binary.")
    parser.add_argument("--bin_w4a8", type=str, default="", help="Path to W4A8 .bin context binary.")
    parser.add_argument("--output_json", type=str, default="benchmark_results_qairt.json", help="Summary results output file.")
    args = parser.parse_args()

    if not os.path.exists(args.classes):
        print(f"[!] Warning: Classes file '{args.classes}' not found.")
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

    os.makedirs(args.work_dir, exist_ok=True)
    raw_input_dir = os.path.join(args.work_dir, "raw_inputs")
    output_run_dir = os.path.join(args.work_dir, "output_runs")
    os.makedirs(raw_input_dir, exist_ok=True)
    os.makedirs(output_run_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(args.test_dir, "*.jpg"))) + sorted(glob.glob(os.path.join(args.test_dir, "*.png")))
    input_list_path = os.path.join(args.work_dir, "input_list.txt")
    valid_samples = []

    print(f"Preparing input tensors from {len(img_files)} images...")
    with open(input_list_path, "w", encoding="utf-8") as flist:
        for idx, img_path in enumerate(img_files):
            base = os.path.splitext(os.path.basename(img_path))[0]
            parts = base.split("_", 2)
            if len(parts) < 3:
                continue
            true_label_raw = parts[2]
            true_idx = resolve_label(true_label_raw, c104_map, synonyms)
            if true_idx is None:
                continue

            raw_path = os.path.join(raw_input_dir, f"input_{idx:04d}.raw")
            if not os.path.exists(raw_path):
                img = Image.open(img_path).convert("RGB").resize((256, 256))
                arr = (np.array(img, dtype=np.float32) / 255.0).astype(np.float32)
                arr.tofile(raw_path)

            flist.write(f"input:={raw_path.replace(chr(92), '/')}\n")
            valid_samples.append((idx, base, true_idx, true_label_raw))

    print(f"Validated {len(valid_samples)} samples for QAIRT evaluation.")

    binaries_to_test = []
    if args.bin_int8 and os.path.exists(args.bin_int8):
        binaries_to_test.append(("MobileNetV4 INT8", args.bin_int8, "W8A8"))
    if args.bin_w4a8 and os.path.exists(args.bin_w4a8):
        binaries_to_test.append(("MobileNetV4 W4A8", args.bin_w4a8, "W4A8"))

    if not binaries_to_test:
        print("[!] No compiled context binaries (.bin) provided or found to benchmark.")
        return

    final_results = {}

    for label, bin_path, fmt in binaries_to_test:
        out_dir = os.path.join(output_run_dir, fmt)
        os.makedirs(out_dir, exist_ok=True)
        
        cmd = [
            args.qairt_bin,
            "--backend", args.backend,
            "--retrieve_context", bin_path,
            "--input_list", input_list_path,
            "--output_dir", out_dir,
            "--perf_profile", "burst"
        ]
        
        print(f"\nRunning benchmark: {label}")
        print(f"Command: {' '.join(cmd)}")
        
        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True)
        total_time = time.perf_counter() - t0
        
        if res.returncode != 0:
            print(f"[ERROR] qnn-net-run failed: {res.stderr}")
            continue
            
        top1_correct = 0
        top3_correct = 0
        total = len(valid_samples)
        
        for i, (sample_idx, sample_name, true_idx, true_label) in enumerate(valid_samples):
            res_file = os.path.join(out_dir, f"Result_{i}", "output.raw")
            if not os.path.exists(res_file):
                continue
            logits = np.fromfile(res_file, dtype=np.float32)
            top_indices = np.argsort(logits)[::-1]
            
            if top_indices[0] == true_idx:
                top1_correct += 1
            if true_idx in top_indices[:3]:
                top3_correct += 1
                
        top1_acc = (top1_correct / total) * 100.0 if total > 0 else 0.0
        top3_acc = (top3_correct / total) * 100.0 if total > 0 else 0.0
        
        final_results[label] = {
            "top1": top1_acc,
            "top3": top3_acc,
            "top1_count": top1_correct,
            "top3_count": top3_correct,
            "total": total,
            "total_time_s": total_time,
            "avg_ms_per_sample": (total_time / total * 1000.0) if total > 0 else 0.0
        }
        
        print(f"RESULTS FOR {label}:")
        print(f"  Top-1 Accuracy : {top1_acc:.2f}% ({top1_correct}/{total})")
        print(f"  Top-3 Accuracy : {top3_acc:.2f}% ({top3_correct}/{total})")

    with open(args.output_json, "w", encoding="utf-8") as fj:
        json.dump(final_results, fj, indent=2)
    print(f"[✓] Benchmark results saved to {args.output_json}")

if __name__ == "__main__":
    main()
