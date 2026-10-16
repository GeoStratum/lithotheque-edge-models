#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automation script for end-to-end AOT compilation of Lithotheque neural models:
- MobileNetV4 (Rock Classification)
- ScaleDetection_MobileNetV4 (Geological Metrology Scale Detection)
- MobileNetV5 (Rock Classification)
"""

import os
import sys
from pathlib import Path
from compile_qairt_aot import setup_env, convert_onnx_to_dlc, quantize_dlc, compile_aot_htp

BASE_DIR = Path(__file__).resolve().parent.parent

def build_all():
    env = setup_env()
    
    models_to_build = [
        {
            "name": "MobileNetV4 (Rock Classification)",
            "onnx_w4": BASE_DIR / "MobileNetV4" / "onnx" / "rock_model_w4a8_MobileNetV4.onnx",
            "onnx_w8": BASE_DIR / "MobileNetV4" / "onnx" / "rock_model_int8_MobileNetV4.onnx",
            "out_dir": BASE_DIR / "MobileNetV4" / "QAIRT",
            "base_w4": "rock_model_w4a8_MobileNetV4",
            "base_w8": "rock_model_w8a8_MobileNetV4"
        },
        {
            "name": "ScaleDetection MobileNetV4 (Scale Metrology)",
            "onnx_w4": BASE_DIR / "ScaleDetection_MobileNetV4" / "onnx" / "v4_small_scale_w4a8.onnx",
            "onnx_w8": BASE_DIR / "ScaleDetection_MobileNetV4" / "onnx" / "v4_small_scale_int8.onnx",
            "out_dir": BASE_DIR / "ScaleDetection_MobileNetV4" / "QAIRT",
            "base_w4": "v4_small_scale_w4a8",
            "base_w8": "v4_small_scale_int8"
        }
    ]
    
    print("\n--- Starting AOT Compilation for All Neural Architectures ---")
    for m in models_to_build:
        print(f"\n==================== Processing {m['name']} ====================")
        
        # W4A8
        if m["onnx_w4"].exists():
            dlc_w4 = m["out_dir"] / f"{m['base_w4']}.dlc"
            convert_onnx_to_dlc(m["onnx_w4"], dlc_w4, env)
            compile_aot_htp(dlc_w4, m["out_dir"], m["base_w4"], env)
            
        # W8A8 / INT8
        if m["onnx_w8"].exists():
            dlc_w8 = m["out_dir"] / f"{m['base_w8']}.dlc"
            convert_onnx_to_dlc(m["onnx_w8"], dlc_w8, env)
            compile_aot_htp(dlc_w8, m["out_dir"], m["base_w8"], env)

    print("\n[OK] All models processed successfully.")

if __name__ == "__main__":
    build_all()
