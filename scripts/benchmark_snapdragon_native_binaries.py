import os
import sys
import time
import glob
import subprocess
import pandas as pd
import numpy as np
from PIL import Image

# Benchmark harness for Qualcomm Snapdragon X Plus Copilot+ PC (Surface Pro 11)
# Fully portable using environment variables or relative paths to current directory
BASE_DIR = os.environ.get("BENCH_WORK_DIR", os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models", "qairt_bin"))
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.join(BASE_DIR, "dataset"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "bench_output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# QAIRT SDK ARM64 Binaries
QAIRT_ROOT = os.environ.get("QAIRT_ARM64_ROOT", os.path.join(BASE_DIR, "QAIRT_SDK", "2.51"))
QAIRT_BIN = os.path.join(QAIRT_ROOT, "bin", "qnn-net-run.exe")
QAIRT_LIB = os.path.join(QAIRT_ROOT, "lib", "aarch64-windows-msvc")
QAIRT_SKEL = os.path.join(QAIRT_ROOT, "lib", "hexagon-v73", "unsigned")
BACKEND = os.path.join(QAIRT_LIB, "QnnHtp.dll")

env = os.environ.copy()
env["PATH"] = f"{QAIRT_LIB};{QAIRT_SKEL};" + env.get("PATH", "")
env["ADSP_LIBRARY_PATH"] = QAIRT_SKEL

image_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.jpg")))[:25]

CONFIGS = [
    {
        "model_name": "MobileNetV4",
        "task": "Rock_Classification",
        "format": "W4A8",
        "bin_file": os.path.join(MODELS_DIR, "rock_model_w4a8_MobileNetV4.bin"),
        "input_name": "input",
        "input_size": (256, 256),
        "precision": "W4A8_INT4_HTP",
        "accuracy": "91.20%"
    },
    {
        "model_name": "MobileNetV4",
        "task": "Rock_Classification",
        "format": "INT8",
        "bin_file": os.path.join(MODELS_DIR, "rock_model_w8a8_MobileNetV4.bin"),
        "input_name": "input",
        "input_size": (256, 256),
        "precision": "W8A8_INT8_HTP",
        "accuracy": "91.80%"
    },
    {
        "model_name": "MobileNetV5",
        "task": "Rock_Classification",
        "format": "W4A8",
        "bin_file": os.path.join(MODELS_DIR, "rock_model_w4a8_MobileNetV5.bin"),
        "input_name": "input",
        "input_size": (512, 512),
        "precision": "W4A8_INT4_HTP",
        "accuracy": "80.33%"
    },
    {
        "model_name": "MobileNetV5",
        "task": "Rock_Classification",
        "format": "INT8",
        "bin_file": os.path.join(MODELS_DIR, "rock_model_w8a8_MobileNetV5.bin"),
        "input_name": "input",
        "input_size": (512, 512),
        "precision": "W8A8_INT8_HTP",
        "accuracy": "83.33%"
    },
    {
        "model_name": "ScaleDetection_MobileNetV4",
        "task": "Scale_Metrology",
        "format": "W4A8",
        "bin_file": os.path.join(MODELS_DIR, "v4_small_scale_w4a8.bin"),
        "input_name": "serving_default_input:0",
        "input_size": (1024, 1024),
        "precision": "W4A8_INT4_HTP",
        "accuracy": "98.45%"
    },
    {
        "model_name": "ScaleDetection_MobileNetV4",
        "task": "Scale_Metrology",
        "format": "INT8",
        "bin_file": os.path.join(MODELS_DIR, "v4_small_scale_int8.bin"),
        "input_name": "serving_default_input:0",
        "input_size": (1024, 1024),
        "precision": "W8A8_INT8_HTP",
        "accuracy": "98.70%"
    }
]

# Run benchmark and output CSVs
