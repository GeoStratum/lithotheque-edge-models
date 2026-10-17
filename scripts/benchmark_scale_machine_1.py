import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# 1. Résolution dynamique et relative des répertoires (BYOM / Portabilité absolue)
BASE_DIR = Path(__file__).resolve().parent.parent

def resolve_models_dir(custom_path: str = None) -> Path:
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)
    env_root = os.environ.get("MODELS_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    # Candidats relatifs prioritaires
    candidates = [
        BASE_DIR / "models",
        BASE_DIR.parent / "fine tuning rock model",
        BASE_DIR.parent / "models",
        BASE_DIR / "ScaleDetection_MobileNetV4" / "onnx",
        BASE_DIR.parent / "fine tuning rock model" / "ScaleDetection_MobileNetV4" / "onnx"
    ]
    for c in candidates:
        if c.exists():
            return c
    return BASE_DIR / "models"

# 2. Support dynamique de cuDNN si installé dans l'environnement virtuel actif
if "VIRTUAL_ENV" in os.environ:
    venv_cudnn = Path(os.environ["VIRTUAL_ENV"]) / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin"
    if venv_cudnn.exists():
        os.environ["PATH"] = str(venv_cudnn) + os.pathsep + os.environ.get("PATH", "")

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not installed. Please install onnxruntime-gpu or onnxruntime.")
    sys.exit(0)

OUTPUT_DIR = BASE_DIR / "results" / "ScaleDetection" / "machine_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = OUTPUT_DIR / "benchmark_machine_1_scale.csv"

def find_scale_model(fmt: str, models_root: Path) -> Path:
    candidates = [
        models_root / f"v4_small_scale_{fmt.lower()}.onnx",
        models_root / "ScaleDetection_MobileNetV4" / "onnx" / f"v4_small_scale_{fmt.lower()}.onnx",
        models_root / "onnx" / f"v4_small_scale_{fmt.lower()}.onnx",
        BASE_DIR / "models" / f"v4_small_scale_{fmt.lower()}.onnx",
        BASE_DIR.parent / "fine tuning rock model" / "ScaleDetection_MobileNetV4" / "onnx" / f"v4_small_scale_{fmt.lower()}.onnx"
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0]

def load_real_images(test_dir: Path, target_shape=(1, 1024, 1024, 3)):
    """
    Charge et prétraite de véritables images réelles (INT8 [-128, 127]) depuis un répertoire de test.
    """
    if not test_dir or not test_dir.exists():
        return []
    
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_files = []
    for ext in extensions:
        image_files.extend(list(test_dir.glob(ext)))
    
    if not image_files:
        return []
    
    preprocessed_tensors = []
    _, h, w, _ = target_shape
    for img_path in sorted(image_files)[:50]:
        try:
            with Image.open(img_path) as pil_img:
                img_rgb = pil_img.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
                arr = np.array(img_rgb, dtype=np.float32)
                # Formule bit-accurate : conversion [0, 255] float vers [-128, 127] signed int8
                arr_int8 = np.clip(np.round(arr - 128.0), -128, 127).astype(np.int8)
                tensor = np.expand_dims(arr_int8, axis=0)
                preprocessed_tensors.append(tensor)
        except Exception as e:
            print(f"[!] Warning: Impossible de charger {img_path.name}: {e}")
            
    return preprocessed_tensors

def main():
    parser = argparse.ArgumentParser(description="Lithotheque Scale Detection ONNX Benchmark (Machine 1)")
    parser.add_argument("--models_dir", type=str, default=None, help="Chemin relatif ou absolu vers les modèles")
    parser.add_argument("--test_dir", type=str, default=None, help="Chemin vers le répertoire d'images réelles de test")
    parser.add_argument("--runs", type=int, default=100, help="Nombre d'itérations d'inférence par backend")
    args = parser.parse_args()

    models_root = resolve_models_dir(args.models_dir)
    print(f"[*] Répertoire modèles résolu : {models_root}")

    # Résolution relative du répertoire de test
    test_dir_path = None
    if args.test_dir:
        test_dir_path = Path(args.test_dir)
    else:
        test_candidates = [
            BASE_DIR / "dataset" / "echelle" / "test",
            BASE_DIR.parent / "fine tuning rock model" / "dataset" / "echelle" / "test",
            BASE_DIR / "test_images"
        ]
        for tc in test_candidates:
            if tc.exists():
                test_dir_path = tc
                break

    real_samples = load_real_images(test_dir_path) if test_dir_path else []
    if real_samples:
        print(f"[*] Jeu de données de test réel chargé : {len(real_samples)} images réelles depuis {test_dir_path}")
    else:
        print("[!] Aucun jeu de test réel trouvé, utilisation de tenseurs de calibration normalisés.")

    input_shape = (1, 1024, 1024, 3)
    configs = [
        ("INT8", "GPU_RTX5060", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        ("W4A8", "GPU_RTX5060", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        ("INT8", "CPU_AVX2", ["CPUExecutionProvider"]),
        ("W4A8", "CPU_AVX2", ["CPUExecutionProvider"])
    ]

    print("\n=== Démarrage du Banc d'Essai ScaleDetection (Machine 1) ===")
    records = []

    for fmt, backend_label, providers in configs:
        model_path = find_scale_model(fmt, models_root)
        if not model_path.exists():
            print(f"[!] Modèle introuvable, passage : {model_path}")
            continue

        print(f"\n--> Évaluation de {fmt} sur {backend_label} (Providers: {providers})")
        sess = ort.InferenceSession(str(model_path), providers=providers)
        active = sess.get_providers()[0]
        print(f"Active Provider: {active}")
        input_name = sess.get_inputs()[0].name

        # Préchauffage (Warmup)
        warmup_input = real_samples[0] if real_samples else np.zeros(input_shape, dtype=np.int8)
        for _ in range(5):
            _ = sess.run(None, {input_name: warmup_input})

        latencies = []
        for i in range(1, args.runs + 1):
            if real_samples:
                inp = real_samples[(i - 1) % len(real_samples)]
            else:
                inp = np.zeros(input_shape, dtype=np.int8)

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

        print(f"Résultats pour {fmt} sur {backend_label}:")
        print(f"  Moyenne: {np.mean(latencies):.2f} ms")
        print(f"  Min:     {np.min(latencies):.2f} ms")
        print(f"  Max:     {np.max(latencies):.2f} ms")
        print(f"  p95:     {np.percentile(latencies, 95):.2f} ms")
        print(f"  FPS:     {1000.0 / np.mean(latencies):.1f} FPS")

    if records:
        df_results = pd.DataFrame(records)
        df_results.to_csv(CSV_OUT, index=False)
        print(f"\n[SUCCÈS] {len(records)} enregistrements de latence empirique sauvegardés dans : {CSV_OUT}")

if __name__ == "__main__":
    main()
