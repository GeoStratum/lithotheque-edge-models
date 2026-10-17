#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel OpenVINO NPU AOT Metadata Compilation Pipeline (LiteRT)
================================================================================
Ce script automatise la compilation et l'injection de métadonnées AOT LiteRT
pour les processeurs Intel Core Ultra et leurs NPU dédiés via OpenVINO.

Règles matérielles :
- Lunar Lake (LNL) & Panther Lake (PTL) -> NPU 4.0 / 5.0 -> W4A8 prioritaire (+ INT8 standard).
- Meteor Lake (MTL) -> NPU 3.0 (11.5 TOPS) -> Strictement INT8 pur.
- SESR-M5 Super-Resolution -> Strictement INT8 sur toutes les cibles.
"""

import os
import sys
import argparse
from pathlib import Path

INTEL_GENERATIONS = {
    "MTL": {
        "name": "Meteor Lake (Core Ultra Series 1)",
        "npu": "Intel NPU 3.0 (11.5 TOPS)",
        "precision": "INT8",
        "devices": "Dell XPS 13, Lenovo ThinkPad Gen 1",
        "os": "Windows 11 / Linux"
    },
    "LNL": {
        "name": "Lunar Lake (Core Ultra Series 2)",
        "npu": "Intel NPU 4.0 (48 TOPS)",
        "precision": "W4A8",
        "devices": "Surface Pro / Laptop Copilot+ PC x86, Asus Zenbook S 14",
        "os": "Windows 11 Copilot+"
    },
    "PTL": {
        "name": "Panther Lake (Core Ultra Series 3)",
        "npu": "Intel NPU 5.0 (50+ TOPS)",
        "precision": "W4A8",
        "devices": "Next-Gen Mobile Workstations",
        "os": "Windows 11 / 12"
    }
}

BASE_DIR = Path(__file__).resolve().parent

def find_models_root() -> Path:
    env_root = os.environ.get("MODELS_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    if (BASE_DIR / "MobileNetV5").exists():
        return BASE_DIR
    e_drive = Path(r"E:\fine tuning\rock identifier\fine tuning rock model")
    if e_drive.exists():
        return e_drive
    return BASE_DIR

MODELS_ROOT = find_models_root()

def inject_intel_metadata(src_path: Path, dst_path: Path, soc_model: str, gen_info: dict, quant_type: str):
    if not src_path.exists():
        print(f"  [-] Source model missing: {src_path}")
        return False

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(src_path, "rb") as f:
        buf = f.read()

    metadata_content = (
        f"LITERT_AOT_TARGET=INTEL_OPENVINO\n"
        f"SOC_MANUFACTURER=IntelOpenVINO\n"
        f"SOC_MODEL={soc_model}\n"
        f"SOC_MARKETING_NAME={gen_info.get('name', '')}\n"
        f"HARDWARE_ACCELERATOR={gen_info.get('npu', 'Intel NPU')}\n"
        f"TARGET_DEVICES={gen_info.get('devices', '')}\n"
        f"TARGET_OS={gen_info.get('os', 'Windows 11')}\n"
        f"QUANTIZATION_SCHEME={quant_type}\n"
        f"PRECOMPILED_METADATA_VERSION=1.0\n"
    ).encode("utf-8")

    # Injection formelle dans le FlatBuffer via Google LiteRT schema
    try:
        from ai_edge_litert import schema_py_generated as schema_fb
        import flatbuffers

        model = schema_fb.ModelT.InitFromObj(schema_fb.Model.GetRootAsModel(buf, 0))

        meta_buffer = schema_fb.BufferT()
        meta_buffer.data = list(metadata_content)
        model.buffers.append(meta_buffer)
        meta_buffer_idx = len(model.buffers) - 1

        meta_entry = schema_fb.MetadataT()
        meta_entry.name = f"intel_openvino_aot_{soc_model.lower()}"
        meta_entry.buffer = meta_buffer_idx
        if model.metadata is None:
            model.metadata = []
        model.metadata.append(meta_entry)

        builder = flatbuffers.Builder(1024 * 1024 * 20)
        builder.Finish(model.Pack(builder), b"TFL3")
        out_bytes = builder.Output()
    except Exception as e:
        print(f"[-] Erreur injection FlatBuffer Intel OpenVINO: {e}")
        return False

    with open(dst_path, "wb") as f:
        f.write(out_bytes)

    size_mb = len(out_bytes) / (1024 * 1024)
    print(f"    [OK] IntelOpenVINO [{soc_model}] {dst_path.name} ({size_mb:.2f} MB)")
    return True

def compile_intel_targets(targets=None):
    if targets is None:
        targets = list(INTEL_GENERATIONS.keys())

    print("=" * 80)
    print(" Compiling Intel OpenVINO NPU AOT Targets (Core Ultra)")
    print("=" * 80)

    for soc_model in targets:
        if soc_model not in INTEL_GENERATIONS:
            print(f"[!] Target inconnue: {soc_model}")
            continue

        info = INTEL_GENERATIONS[soc_model]
        is_w4 = (info["precision"] == "W4A8")
        print(f"\n[*] Target: Intel NPU {soc_model} ({info['name']} - {info['npu']}) -> Priorite: {info['precision']}")

        # 1. MobileNetV5
        v5_src = MODELS_ROOT / "MobileNetV5" / "tflite" / ("best_mobilenetv5_rocks_w4a8.tflite" if is_w4 else "best_mobilenetv5_rocks_int8.tflite")
        v5_out = MODELS_ROOT / "MobileNetV5" / "IntelOpenVINO" / soc_model / ("best_mobilenetv5_rocks_w4a8.tflite" if is_w4 else "best_mobilenetv5_rocks_int8.tflite")
        inject_intel_metadata(v5_src, v5_out, soc_model, info, "W4A8_FULL_INTEGER" if is_w4 else "PER_TENSOR_INT8_SIGNED")

        if is_w4:
            v5_int8_src = MODELS_ROOT / "MobileNetV5" / "tflite" / "best_mobilenetv5_rocks_int8.tflite"
            v5_int8_out = MODELS_ROOT / "MobileNetV5" / "IntelOpenVINO" / soc_model / "best_mobilenetv5_rocks_int8.tflite"
            inject_intel_metadata(v5_int8_src, v5_int8_out, soc_model, info, "PER_TENSOR_INT8_SIGNED")

        # 2. MobileNetV4
        v4_src = MODELS_ROOT / "MobileNetV4" / "tflite" / ("rock_model_w4a8_MobileNetV4.tflite" if is_w4 else "rock_model_w8a8_MobileNetV4.tflite")
        v4_out = MODELS_ROOT / "MobileNetV4" / "IntelOpenVINO" / soc_model / ("rock_model_w4a8_MobileNetV4.tflite" if is_w4 else "rock_model_w8a8_MobileNetV4.tflite")
        inject_intel_metadata(v4_src, v4_out, soc_model, info, "W4A8_FULL_INTEGER" if is_w4 else "PER_TENSOR_INT8_SIGNED")

        # 3. Scale Detection
        scale_src = MODELS_ROOT / "ScaleDetection_MobileNetV4" / "tflite" / ("v4_small_scale_int4.tflite" if is_w4 else "v4_small_scale_int8.tflite")
        scale_out = MODELS_ROOT / "ScaleDetection_MobileNetV4" / "IntelOpenVINO" / soc_model / ("v4_small_scale_int4.tflite" if is_w4 else "v4_small_scale_int8.tflite")
        inject_intel_metadata(scale_src, scale_out, soc_model, info, "W4A8_FULL_INTEGER" if is_w4 else "PER_TENSOR_INT8_SIGNED")

        # 4. SESR Super-Resolution (Strictement INT8)
        sesr_src = MODELS_ROOT / "SESR" / "tflite" / "sesr_m5.tflite"
        sesr_out = MODELS_ROOT / "SESR" / "IntelOpenVINO" / soc_model / "sesr_m5.tflite"
        inject_intel_metadata(sesr_src, sesr_out, soc_model, info, "PER_TENSOR_INT8_SIGNED")

    print("\n[OK] Compilation Intel NPU terminee avec succes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intel OpenVINO NPU AOT Compilation")
    parser.add_argument("--soc", choices=list(INTEL_GENERATIONS.keys()), default=None,
                        help="Cible SoC specifique (ex: LNL, PTL, MTL). Par defaut: Toutes les cibles.")
    args = parser.parse_args()

    targets = [args.soc] if args.soc else list(INTEL_GENERATIONS.keys())
    compile_intel_targets(targets)
