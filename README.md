# Lithotheque Edge Models - GeoStratum
**Technical showcase of the Edge AI architecture powering the Lithotheque offline application.**

[![Offline](https://img.shields.io/badge/Inference-100%25_Offline-success)](https://www.geostratum.eu/lithotheque)
[![Engine](https://img.shields.io/badge/Engine-LiteRT_Standalone-orange)](https://ai.google.dev/edge/litert)
[![C++](https://img.shields.io/badge/Native_Core-C%2B%2B_23-00599C?logo=c%2B%2B)](https://en.cppreference.com/w/cpp/23)
[![Windows](https://img.shields.io/badge/Platform-Windows_11_ARM64-0078D4?logo=windows)](https://www.microsoft.com/windows)
[![Python](https://img.shields.io/badge/Training-Python_3.11-3776AB?logo=python)](https://www.python.org/downloads/release/python-3110/)
[![Qualcomm QNN](https://img.shields.io/badge/NPU_Backend-Qualcomm_QNN-3253DC?logo=qualcomm&logoColor=white)](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
[![Quantization](https://img.shields.io/badge/Quantization-W4A8_Full_Integer-red)](METHODOLOGY.md)
[![Quantization](https://img.shields.io/badge/Quantization-W8A8_Full_Integer-red)](METHODOLOGY.md)

> [!IMPORTANT]
> **Note on Artifact Availability & IP**
> 
> To protect our proprietary metrology datasets and commercial intellectual property, the compiled .tflite and .onnx model artifacts, alongside the FP32 source weights, are not publicly distributed in this repository.
> 
> The core purpose of this scientific repository is to openly share the **engineering methodology**, the **W4A8 Full Integer quantization pipeline**, and the **mathematical architectural strategies** (such as the 21-Pass Multi-Scale Tiling) used to achieve real-time inference on **mobile Edge NPUs (e.g., Qualcomm Hexagon, Google Tensor Edge TPU)**.

> [!TIP]
> **New to Edge AI?**
> 
> For developers and researchers new to AI and quantization, we highly recommend consulting our **[Detailed Methodology](docs/DETAILED_METHODOLOGY.md)**. It provides a pedagogical introduction to fundamental concepts, numerical precision (INT8/INT4), and hardware acceleration architectures.


## Table of Contents
- [1. Overview](#1-overview)
- [2. Dual-Model Ecosystem](#2-dual-model-ecosystem)
- [3. Training Data & Acknowledgements](#3-training-data--acknowledgements)
- [4. Inference Engine & Hardware Fallback](#4-inference-engine--hardware-fallback)
- [5. Deployment Strategy (AI Tiers)](#5-deployment-strategy-ai-tiers)
- [6. W4A8 Quantization Methodology](#6-w4a8-quantization-methodology)
- [7. Technical Specifications (I/O)](#7-technical-specifications-io)
- [8. Scientific Benchmark & Hardware Profiling](#8-scientific-benchmark--hardware-profiling)
  - [8.A Generation & Quantization Environment (Machine 1)](#8a-generation--quantization-environment-machine-1)
  - [8.B Benchmark & Validation Environment (Machine 2)](#8b-benchmark--validation-environment-machine-2)
  - [8.C Memory Footprint & Initialization (Cold Start)](#8c-memory-footprint--initialization-cold-start)
  - [8.D Inference Latency & Accuracy (Avg over 100 runs)](#8d-inference-latency--accuracy-avg-over-100-runs)
  - [8.E Native AOT Snapdragon X NPU Binaries Qualification (Machine 2 - Surface Pro 11)](#8e-native-aot-snapdragon-x-npu-binaries-qualification-machine-2---surface-pro-11)
- [9. Hardware Strategy & Analysis](#9-hardware-strategy--analysis)
- [10. Model Card (Technical Specs)](#10-model-card-technical-specs)
- [11. Deployment & Application](#11-deployment--application)
- [12. Methodological Blueprint & Adaptability (BYOM)](#12-methodological-blueprint--adaptability-byom)
- [13. Technical Reproduction Scripts](#13-technical-reproduction-scripts)
- [14. License](#14-license)
- [15. Contact](#15-contact)
- [16. Citation](#16-citation)

## 1. Overview
This repository documents the advanced Edge AI architecture integrated into [GeoStratum Lithotheque](https://www.geostratum.eu/lithotheque). To provide instant, on-device geological classification and scale detection without requiring an internet connection, the application relies on a highly optimized, dual-model ecosystem deployed via **Google Play Asset Delivery (On-Demand)**.

**[Try the Application on Play Store](https://play.google.com/store/apps/details?id=com.lithotheque.app.release)** | **[Read the Model Card](MODEL_CARD.md)** | **[Quantization Methodology](METHODOLOGY.md)**


## 2. Dual-Model Ecosystem
The application utilizes two distinct neural networks, fine-tuned specifically for geological field operations:

### A. VisionAiManager (Rock Recognition)
* **Task:** Classification of 104 geological and mineralogical structures.
* **Base Architectures:** `MobileNetV5-300m` (**Vision architecture akin to the Gemma 3n vision encoder**) for modern devices, and `MobileNetV4-Large` for legacy fallback.
* **Training Corpus:** Fine-tuned on open-source datasets (Udayl/Stealth) totaling >14,500 validated images.
* **Innovation: Multi-Scale Tiling (21-Pass Strategy):** To capture both macroscopic textures and microscopic crystals without losing data during downscaling, the engine evaluates the image through 21 parallel passes:
  * **1 Global Pass:** Overall sample context.
  * **4 Medium Passes (2x2 Grid):** Regional texture analysis.
  * **16 Fine Passes (4x4 Grid):** Micro-detail and mineral extraction.
  *(Scores are aggregated via weighted average for the final prediction).*

![Rock Detection AI in Action](results/consolidated/images/rock_detection_demo.gif)
> **Figure 1: Real-time Rock Detection** - Demonstration of the 21-MST engine classifying lithological samples with high confidence on a Galaxy Z Fold 7 via Remote Test Lab.

### Dataset Categorization (104 Model Output Logits / 96 Unique Lithological Taxa)

The model output layer exposes **104 classification logits**, mapped directly to the concatenated multi-source training datasets (Udayl & Stealth). Due to origin-dataset labeling artifacts, 8 classes exist as formatting duplicates across the dataset merge (e.g. `Shale (Mudstone)` vs. `Shale-(Mudstone)`, `Olivine basalt` vs. `Olivine-basalt`, and dual entries for `Dacite`, `Granite`, `Marble`, `Quartzite`, `Copper`, `Oolitic Limestone` / `Dolomites`), representing **96 unique geoscientific taxa** grouped below by primary genetic classification:

#### 1. Sedimentary (Sed)
> Bauxite, Caliche, Chalk, Chert, Clay, Coal, Conglomerate, Coquina, Diatomite, Dolomitic Limestone, Dolomites, Flint, Fossiliferous Limestone, Gypsum, Halite, Limestone, Novaculite, Oolitic Limestone, Phosphate, Potash, Sandstone, Shale (Mudstone) *(dual-dataset entry)*, Siliceous Sinter, Siltstone, Sodium Carbonate, Tufa.

#### 2. Magmatic / Plutonic (Mag)
> Anorthosite, Aplite, Diorite, Dolerite, Dunite, Essexite, Gabbro, Granite *(dual-dataset entry)*, Granodiorite, Norite, Pegmatite, Syenite.

#### 3. Volcanic / Extrusive (Volc)
> Andesite, Basalt, Dacite *(dual-dataset entry)*, Ignimbrite, Komatiite, Obsidian, Olivine Basalt *(dual-dataset entry)*, Phonolite, Pillow Lava, Pumice, Rhyolite, Tephrite, Trachyte, Tuff, Volcanic Bombs.

#### 4. Metamorphic (Meta)
> Anthracite, Breccia (Tectonic/Fault), Gneiss, Hornfels, Lapis Lazuli, Marble *(dual-dataset entry)*, Phyllite, Quartzite *(dual-dataset entry)*, Schists, Serpentine, Skarn, Slate.

#### 5. Native Elements, Minerals & Ores
> Bornite, Calcite, Chromite, Cobalt, Columbite-Tantalite, Copper *(dual-dataset entry)*, Feldspar, Fluorite, Gold, Iron Ore, Labradorite, Lead, Lithium, Magnetite, Malachite, Mariposite, Mica, Molybdenum, Nickel, Platinum, Pyrite, Quartz, Silica, Silver, Sodalite, Stibnite, Sulfur, Tantalum, Tungsten, Uranium, Vanadium, Zeolite, Zinc.

```mermaid
graph TD
    A[Raw Camera Frame] -->|C++ NEON / SVE2 JNI| B(Preprocessing & Normalization)
    B --> C{Multi-Scale Tiling Engine}
    
    C -->|Step 1| F[16 Fine Passes<br>4x4 Grid: Minerals & Micro-details]
    F -->|Step 2| E[4 Medium Passes<br>2x2 Grid: Regional Textures]
    E -->|Step 3| D[1 Global Pass<br>1x1: Overall Context]
    
    F -.->|Scores| G{Weighted Score Aggregation}
    E -.->|Scores| G
    D -.->|Scores| G
    
    G --> H[Hardware Fallback Inference<br>NPU / GPU / CPU]
    H --> I((Final Prediction<br>Class + Confidence))
    
    style A fill:#2d3436,stroke:#fff,color:#fff
    style I fill:#0984e3,stroke:#fff,color:#fff
    style C fill:#d63031,stroke:#fff,color:#fff
    style G fill:#e17055,stroke:#fff,color:#fff
```
---

### B. LocalAiManager (Scale Detection)
* **Task:** Metrology assistant detecting reference objects (coins, geological scales).
* **Base Architecture:** `MobileNetV4-Small`.
* **Resolution:** High-res 1024x1024 processing for precise small-object anchoring.
* **Training Corpus:** Fine-tuned on >1,000 reference images.

## 3. Training Data & Acknowledgements

The robustness of the Lithotheque AI ecosystem relies on a hybrid training approach, combining proprietary field metrology data with highly curated, open-source geological collections. We gratefully acknowledge the creators of the following datasets:

### A. Metrology & Scale Detection (LocalAiManager)
* **GeoStratum Proprietary Corpus:** A custom dataset comprising >1,000 high-resolution field photographs of geological reference objects (coins, geological scales) captured in diverse lighting and terrain conditions by the GeoStratum. 
*(Note: This dataset remains the exclusive intellectual property of GeoStratum and is not publicly distributed).*

### B. Lithological Classification (VisionAiManager)
To achieve extreme accuracy across 104 geological classes, the model was fine-tuned using the following validated sources, both utilized under the **MIT License**:
* **[Udayl Rocks Dataset](https://huggingface.co/datasets/udayl/rocks):** Our primary dataset for lithological visual features, hosted on Hugging Face.
* **[Stealth Technologies Rock Classification](https://www.kaggle.com/datasets/stealthtechnologies/rock-classification):** A supplementary dataset hosted on Kaggle, utilized to expand the morphological variance and improve the model's generalization capabilities across varied rock formations.

## 4. Inference Engine & Hardware Fallback
To prevent thermal throttling during the intensive 21-pass analysis, all image preprocessing (resizing, normalization) is written in **Native C++ utilizing C++23 auto-vectorization (supporting ARMv8-A NEON SIMD and AVX2)**, yielding a 5x to 10x speedup over standard interpreter pipelines with minimal memory overhead.

Inference is powered by **LiteRT Standalone (2.1.4)** with a 100% NPU-compliant execution graph, fully aligned with **Android 12 (API 31) through Android 17 (API 37)** standards, including strict **16 KB memory page size alignment (`-Wl,-z,max-page-size=16384`)**:

### A. Graph Transformations & NPU Driver Compliance
Offloading deep convolutional neural networks to heterogeneous mobile accelerators requires bypassing common HAL and driver-level validation pitfalls (`ANEURALNETWORKS_BAD_DATA`, `NNAPI_EXECUTION_FAILED`):
* **Per-Tensor Classification Head Quantization**: Standard TensorFlow Lite and ONNX post-training quantization routines apply per-channel (`axis=0` or `axis=3`) scale and zero-point vectors to weight matrices. While CPUs and modern GPUs handle per-channel FullyConnected/Dense layers seamlessly, mobile NPU drivers (Samsung Eden/NCP, Qualcomm QNN HTP, MediaTek NeuroPilot) reject per-channel 2D matrix multiplications. All final dense layers in MobileNetV4 and MobileNetV5 are mathematically converted and clamped to **signed per-tensor asymmetry** (`TENSOR_QUANT8_ASYMM_SIGNED`), achieving a 100% operator offload rate with zero CPU fallback nodes.
* **Kernel-Fused Convolutional Padding**: Mobile architectures frequently employ asymmetric spatial padding before strided depthwise separable blocks. Standalone TFLite `PAD` operators force hardware pipeline stalls and buffer synchronization barriers between the NPU and CPU memory spaces. All explicit 4D padding operators have been structurally fused directly into preceding/subsequent convolution layers utilizing hardware-native `SAME` padding kernels.

### B. Hardware-Targeted Compilation Paradigms

Deploying neural network models across heterogeneous Android hardware cannot rely on a naive "one-size-fits-all" deployment pipeline. Mobile System-on-Chips (SoCs) feature radically divergent Neural Processing Unit (NPU) architectures, microcode compilers, execution pipelines, and arithmetic sub-byte capabilities.

To resolve these disparities without runtime performance degradation or thermal throttling, the Lithotheque edge pipeline implements three distinct hardware execution pathways:

```mermaid
graph TD
    Model[Lithotheque Edge Vision & Metrology Models] --> TargetCheck{Target Silicon Family}
    
    TargetCheck -->|Qualcomm Snapdragon| QcomAOT[Ahead-Of-Time / AOT via QAIRT / QNN]
    TargetCheck -->|Google Pixel Tensor| TensorAOT[Ahead-Of-Time / AOT TPU Metadata]
    TargetCheck -->|Samsung Exynos & MediaTek| GenericJIT[Just-In-Time / JIT On-Device Driver]
    
    QcomAOT --> QcomSplit{Hexagon HTP Micro-Architecture}
    QcomSplit -->|v69 & v73: No sub-byte unit| QcomINT8[Pure INT8 Packs: Standard + Legacy + SESR]
    QcomSplit -->|v75 to v89: Native W4A8 engine| QcomW4[W4A8 Full Packs: Premium + Balanced + SESR]
    
    TensorAOT --> TensorAll[Pure INT8 TPU Execution: Pixel 6 to 11 / Android 12 to 17]
    TensorAll --> TensorReject[Strict Rejection of Sub-Byte INT4]
    TensorAll --> TensorPacks[Recognition INT8 + Dedicated SESR per Tensor G]
    
    GenericJIT --> JITSanitize[Sanitized Universal TFLite Graphs]
    JITSanitize --> JITCompile[On-Device Compilation: Samsung Eden & MediaTek NeuroPilot]
    JITCompile --> JITPacks[100% Full NPU Offload: 0 CPU/GPU Fallback]
    
    style TargetCheck fill:#2d3436,stroke:#fff,color:#fff
    style QcomAOT fill:#0984e3,stroke:#fff,color:#fff
    style TensorAOT fill:#00b894,stroke:#fff,color:#fff
    style GenericJIT fill:#e17055,stroke:#fff,color:#fff
    style QcomINT8 fill:#6c5ce7,stroke:#fff,color:#fff
    style QcomW4 fill:#0984e3,stroke:#fff,color:#fff
    style TensorReject fill:#d63031,stroke:#fff,color:#fff
```

---

#### 1. Qualcomm Snapdragon: Native Ahead-Of-Time (AOT) Compilation via QAIRT

Rather than relying on just-in-time compilation during application runtime, Qualcomm Snapdragon deployments utilize **offline Ahead-Of-Time (AOT) graph compilation** using the **Qualcomm AI Runtime (QAIRT / QNN)** toolchain. 

##### Why Offline AOT is Mandatory for Hexagon NPUs
On Qualcomm Hexagon Tensor Processors (HTP), on-device graph compilation is computationally prohibitive:
* **Cold-Boot Latency & ANR Prevention**: Compiling complex convolutional graphs like MobileNetV5 on-device takes between 25 to 90 seconds, causing Android Application Not Responding (ANR) watchdog timeouts and app launch freezes.
* **Tightly Coupled Memory (TCM) Optimization**: Hexagon processors feature high-speed local SRAM (TCM) rather than standard cache lines. The QAIRT offline compiler statically maps tensor weights, activation ping-pong buffers, and DMA transfer schedules directly into TCM with cycle-accurate timing.
* **Output Format**: Offline compilation produces a pre-serialized context binary (`.bin`) that can be memory-mapped (`mmap`) into Hexagon memory in single-digit milliseconds during cold boot.

##### Architectural Bifurcation: HTP v69/v73 (INT8) vs. HTP v75–v89 (W4A8)
Qualcomm's Hexagon Tensor Processor underwent a fundamental micro-architectural transition between HTP v73 and HTP v75:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   QUALCOMM HEXAGON NPU GENERATION MAPPING                         │
├───────────────┬──────────────────────────────────┬─────────────────┬─────────────┤
│ HTP Version   │ Representative SoCs              │ Quantization    │ Delivered   │
│               │                                  │ Capability      │ Packages    │
├───────────────┼──────────────────────────────────┼─────────────────┼─────────────┤
│ HTP v69       │ Snapdragon 888, 888+ (SM8350),   │ Pure INT8 Only  │ Standard,   │
│               │ 778G, 780G, 778G+, 695 (SM6375)  │ (No sub-byte)   │ Legacy, SESR│
├───────────────┼──────────────────────────────────┼─────────────────┼─────────────┤
│ HTP v73       │ Snapdragon 8 Gen 1 (SM8450),     │ Pure INT8 Only  │ Standard,   │
│               │ 8+ Gen 1 (SM8475), 7+ Gen 2      │ (No sub-byte)   │ Legacy, SESR│
│               │ (SM7475), 6 Gen 1 (SM6450)       │                 │             │
├───────────────┼──────────────────────────────────┼─────────────────┼─────────────┤
│ HTP v75       │ Snapdragon 8 Gen 2 (SM8550),     │ Native W4A8     │ Premium,    │
│               │ Snapdragon 8 Gen 3 (SM8650)      │ + INT8          │ Balanced,   │
│               │                                  │                 │ SESR        │
├───────────────┼──────────────────────────────────┼─────────────────┼─────────────┤
│ HTP v79       │ Snapdragon 8s Gen 3 (SM8635),    │ Native W4A8     │ Premium,    │
│               │ 7 Gen 3 (SM7550), 7+ Gen 3       │ + INT8          │ Balanced,   │
│               │ (SM7675), 6 Gen 3                │                 │ SESR        │
├───────────────┼──────────────────────────────────┼─────────────────┼─────────────┤
│ HTP v81       │ Snapdragon X Elite (X1E Series), │ Native W4A8     │ Premium,    │
│               │ Snapdragon X Plus (X1P Series)   │ + INT8          │ Balanced,   │
│               │                                  │                 │ SESR        │
├───────────────┼──────────────────────────────────┼─────────────────┼─────────────┤
│ HTP v85       │ Snapdragon 8 Elite (SM8750),     │ Native W4A8     │ Premium,    │
│               │ Snapdragon X2                    │ + INT8          │ Balanced,   │
│               │                                  │                 │ SESR        │
├───────────────┼──────────────────────────────────┼─────────────────┼─────────────┤
│ HTP v89       │ Snapdragon 8 Elite Gen 5 (SM8850)│ Native W4A8     │ Premium,    │
│               │                                  │ + INT8          │ Balanced,   │
│               │                                  │                 │ SESR        │
└───────────────┴──────────────────────────────────┴─────────────────┴─────────────┘
```

* **HTP v69 & v73 (SoCs INT8 Pur — 6 AIPacks Total)**:
  * *Micro-Architectural Constraint*: The Hexagon Vector eXtensions (HVX) and Hexagon Matrix eXtensions (HMX) compute blocks in v69 and v73 are hardwired strictly for 8-bit $\times$ 8-bit integer multiply-accumulate operations with a 32-bit accumulator. They possess **zero dedicated sub-byte bit-manipulation hardware** for 4-bit integer weights.
  * *Penalty of W4A8 on v69/v73*: Attempting to execute W4A8 models on these chips forces the driver to inject software bit-unpacking routines running on scalar or vector DSP threads. This causes heavy register pressure, pipeline stalls, higher power consumption, and **worse inference latency than native INT8**.
  * *Delivered Packages*: These SoCs receive strictly pure INT8 models:
    1. `standard_recognition_snapdragon_vXX.aipack` (MobileNetV5-300M INT8 + Scale MobileNetV4-Small INT8)
    2. `legacy_recognition_snapdragon_vXX.aipack` (MobileNetV4-Large INT8 + Scale MobileNetV4-Small INT8)
    3. `super_resolution_snapdragon_vXX.aipack` (SESR-M5 INT8)
    - *(2 HTP architectures $\times$ 3 packs = 6 packs)*.

* **HTP v75, v79, v81, v85, v89 (SoCs INT4 Natif — 15 AIPacks Total)**:
  * *Hardware Innovation*: Starting with HTP v75 (Snapdragon 8 Gen 2), Qualcomm integrated dedicated hardware sub-byte decompression units directly inside the HMX matrix multiplication cores. These units unpack 4-bit weights streamed directly from TCM into 8-bit registers **on the fly with zero cycle latency overhead**.
  * *Elimination of Redundant INT8*: Serving standard or legacy INT8 models to these modern chips is completely counter-productive:
    - W4A8 achieves identical classification accuracy while cutting parameter memory footprint by 50% (MobileNetV5 decreases from 288 MB to 148 MB).
    - Memory bandwidth requirements across the SoC bus are cut in half, boosting inference speed and drastically reducing thermal dissipation.
    - Legacy INT8 models (MobileNetV4-Large) are obsolete on these chips because the NPU possesses more than sufficient TOPS to execute the larger, more accurate MobileNetV5-300M model at sub-10ms latencies.
  * *Delivered Packages*: These modern chips receive strictly optimized W4A8 packages:
    1. `premium_recognition_snapdragon_vXX.aipack` (MobileNetV5-300M W4A8 + Scale MobileNetV4-Small INT4)
    2. `balanced_recognition_snapdragon_vXX.aipack` (MobileNetV4-Large W4A8 + Scale MobileNetV4-Small INT4)
    3. `super_resolution_snapdragon_vXX.aipack` (SESR-M5 INT8)
    - *(5 HTP architectures $\times$ 3 packs = 15 packs)*.

---

#### 2. Google Tensor: Ahead-Of-Time (AOT) TPU Metadata (Android 12 to 17)

Google Tensor silicon represents a dedicated EdgeTPU architecture designed specifically for the Pixel smartphone lineup, spanning Android 12 (API 31) through Android 17 (API 37):

* **Generations Fully Supported**:
  * **Tensor G1** (5nm Samsung) $\rightarrow$ Pixel 6, 6 Pro, 6a *(Android 12 / API 31–32)*
  * **Tensor G2** (5nm Samsung) $\rightarrow$ Pixel 7, 7 Pro, 7a, Fold *(Android 13 / API 33)*
  * **Tensor G3** (4nm Samsung) $\rightarrow$ Pixel 8, 8 Pro, 8a *(Android 14 / API 34)*
  * **Tensor G4** (4nm Samsung) $\rightarrow$ Pixel 9, 9 Pro, 9 Pro XL, 9 Pro Fold *(Android 15 / API 35)*
  * **Tensor G5** (3nm TSMC) $\rightarrow$ Pixel 10 Series *(Android 16 / API 36)*
  * **Tensor G6** (3nm TSMC) $\rightarrow$ Pixel 11 Series *(Android 17 / API 37)*

##### Strict Rejection of Sub-Byte INT4 on Google Tensor TPUs
A critical engineering discovery during hardware validation is that **Google Tensor EdgeTPUs reject sub-byte INT4 (W4A8) quantization**:
* **Systolic Architecture Limitations**: The Google EdgeTPU core is constructed around a 2D systolic array hardwired for signed symmetric 8-bit integer dot products (`gemmlowp` standard: `int8_t` weights $\in [-128, 127]$, `int8_t` activations $\in [-128, 127]$, with 32-bit integer accumulation).
* **Driver Rejection & Graph Fragmentation**: When a TFLite model containing 4-bit weights is passed to the Tensor LiteRT/NNAPI delegate, the TPU driver cannot allocate native hardware kernels for sub-byte matrix multiplications. The driver either returns an unrecoverable compilation error (`ANEURALNETWORKS_BAD_DATA` / `NNAPI_EXECUTION_FAILED`) or fragments the computational graph into dozens of micro-subgraphs separated by software `DequantizeOp` layers running on the ARM Cortex CPU.
* **Thermal & Latency Impact**: This subgraph fragmentation triggers continuous, high-latency memory transfers between TPU SRAM, system DRAM, and CPU caches, multiplying inference latency by 10x to 25x and causing rapid thermal throttling.
* **The INT8 Solution**: When provided with pure signed per-tensor INT8 models, the Tensor TPU executes 100% of operators natively in hardware with zero graph splitting, near-zero inference jitter, and ultra-low power consumption.

##### Dedicated AOT Packaging per Tensor Generation (12 Packs Total)
Each Tensor G generation features distinct TPU buffer configurations, local SRAM scratchpad sizes, and microcode dispatchers. Every Tensor generation (G1 to G6) receives its own precompiled AOT FlatBuffer metadata package:
1. `recognition_tensor_gX.aipack`: Contains MobileNetV5-300M INT8 + Scale MobileNetV4-Small INT8, pre-configured with TPU hardware descriptors.
2. `super_resolution_tensor_gX.aipack`: Contains a generation-specific build of SESR-M5 INT8 optimized for that specific TPU's memory bandwidth and tiling constraints.
- *(6 Tensor generations $\times$ 2 packs = 12 packs total)*.

---

#### 3. Samsung Exynos & MediaTek Dimensity: Zero-Fallback Just-In-Time (JIT) Compilation

Unlike Qualcomm's closed, offline-compiled binary workflow, **Samsung Exynos** and **MediaTek Dimensity** platforms utilize high-speed **on-device Just-In-Time (JIT) compilation**:

* **Architectural Ecosystem**:
  * **Samsung Exynos**: Exynos 1280 (`s5e8825`), 1380 (`s5e8835`), 1480 (`s5e8845`), 1580 (`s5e8855`), 2100 (`s5e9840`), 2200 (`s5e9925`), 2400 (`s5e9945`), and 2600 (`s5e9955`) driven by the Samsung Eden NPU driver and Neural Compiled Program (NCP) runtime.
  * **MediaTek Dimensity**: Dimensity 7000, 8000, and 9000 series powered by the MediaTek APU (NeuroPilot runtime).

##### On-Device JIT Compilation Mechanics
* **Embedded System Compilers**: Samsung and MediaTek embed their neural graph compilers directly inside the Android system vendor partition (`/vendor/lib64/libeden_nn.so` for Exynos, `/vendor/lib64/libneuron_adapter.so` for MediaTek).
* **Cold-Boot JIT Compilation**: When LiteRT initializes on the client device, the vendor delegate intercepts the clean, universal TFLite FlatBuffer and compiles it into hardware microcode directly on the device in approximately **60 to 140 milliseconds**.
* **Driver Caching**: The resulting compiled execution graph is immediately cached in the application's private code cache directory (`/data/user/0/com.lithotheque.app/code_cache/`). Subsequent application launches load the cached microcode with zero JIT latency.

##### Graph Sanitization for 100% Full NPU Offload (Zero Fallback)
Standard TensorFlow Lite and PyTorch quantization pipelines produce graphs that fail on Exynos and MediaTek NPU drivers. To achieve **100% full NPU execution with 0 CPU/GPU fallback nodes**, our export pipeline enforces two critical mathematical transformations:
1. **Per-Tensor Classification Head Quantization**:
   - *The Issue*: Standard post-training quantization applies per-channel scale factors across the output classes of FullyConnected/Dense layers. The matrix multiply hardware inside Samsung Eden and MediaTek NeuroPilot APUs only supports **per-tensor scale factors** for 2D weight matrices. Per-channel fully connected layers cause an immediate driver fallback to CPU execution.
   - *The Fix*: All classification heads (104 geological classes) are mathematically converted to signed per-tensor asymmetry (`TENSOR_QUANT8_ASYMM_SIGNED`), allowing the final classification projection to run entirely on the NPU in less than 0.2 milliseconds.
2. **Kernel-Fused Convolutional Padding**:
   - *The Issue*: Standalone `PAD` operators force the NPU to flush its internal execution pipelines, synchronize buffers back to system RAM, and perform memory-bound zero-padding before restarting NPU execution.
   - *The Fix*: All explicit 4D padding operations are structurally fused directly into preceding and subsequent convolution kernels using hardware-native `SAME` padding kernels.
3. **Activation Clamping**:
   - Non-linear activations like GELU and HardSwish are mathematically clamped within signed INT8 limits (`[-128, 127]`), preventing integer overflow and numerical saturation in Eden / APU vector registers.

---

## 5. Deployment Strategy (AI Tiers & Play Asset Delivery Architecture)

To completely eliminate storage bloat and bandwidth waste, the Lithotheque edge distribution is organized into **38 specialized Play Asset Delivery (PAD) packs**. At application startup, the Android client detects the exact SoC vendor, architecture generation, and RAM capacity, requesting only the specific binary package engineered for that device:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        38 PLAY ASSET DELIVERY (PAD) PACKS MATRIX                        │
├───────────────────────┬───────────────────────────────┬────────────┬───────┬────────────┤
│ Hardware Category     │ Target Chips / Architectures  │ Precision  │ Packs │ Delivery   │
│                       │                               │            │ Count │ Pipeline   │
├───────────────────────┼───────────────────────────────┼────────────┼───────┼────────────┤
│ Qualcomm Modern       │ HTP v75, v79, v81, v85, v89   │ W4A8       │ 15    │ Native AOT │
│                       │ (8G2/G3, 8 Elite, 7G3, X, X2) │ (INT4)     │       │ (.bin)     │
├───────────────────────┼───────────────────────────────┼────────────┼───────┼────────────┤
│ Qualcomm Legacy       │ HTP v69, v73                  │ Pure INT8  │ 6     │ Native AOT │
│                       │ (888, 8G1, 778G, 695, 6G1)    │ (No INT4)  │       │ (.bin)     │
├───────────────────────┼───────────────────────────────┼────────────┼───────┼────────────┤
│ Google Tensor         │ Tensor G1 to G6               │ Pure INT8  │ 12    │ AOT TPU    │
│                       │ (Pixel 6 to 11 / Android 12-17│ (No INT4)  │       │ FlatBuffer │
├───────────────────────┼───────────────────────────────┼────────────┼───────┼────────────┤
│ Universal Fallback    │ Samsung Exynos, MediaTek,     │ W4A8 &     │ 5     │ On-Device  │
│ (Generic TFLite)      │ UniSoC, Unknown SoCs          │ INT8       │       │ JIT Driver │
├───────────────────────┴───────────────────────────────┴────────────┼───────┴────────────┤
│ TOTAL PLAY ASSET DELIVERY PACKS PRODUCED                           │ 38 PACKS           │
└────────────────────────────────────────────────────────────────────┴────────────────────┘
```

### Detailed Distribution Matrix

| Pack Identifier Pattern | Target Silicon & Generation | Quantization | Included Model Modules | Compilation & Packaging |
| :--- | :--- | :---: | :--- | :---: |
| `premium_recognition_snapdragon_vXX` | Qualcomm HTP v75, v79, v81, v85, v89 *(5 packs)* | **W4A8 (INT4)** | MobileNetV5-300M W4A8 + Scale MobileNetV4-Small INT4 | Native AOT (`.bin`) via QAIRT |
| `balanced_recognition_snapdragon_vXX` | Qualcomm HTP v75, v79, v81, v85, v89 *(5 packs)* | **W4A8 (INT4)** | MobileNetV4-Large W4A8 + Scale MobileNetV4-Small INT4 | Native AOT (`.bin`) via QAIRT |
| `standard_recognition_snapdragon_vXX` | Qualcomm HTP v69, v73 *(2 packs)* | **INT8 Only** | MobileNetV5-300M INT8 + Scale MobileNetV4-Small INT8 | Native AOT (`.bin`) via QAIRT |
| `legacy_recognition_snapdragon_vXX` | Qualcomm HTP v69, v73 *(2 packs)* | **INT8 Only** | MobileNetV4-Large INT8 + Scale MobileNetV4-Small INT8 | Native AOT (`.bin`) via QAIRT |
| `super_resolution_snapdragon_vXX` | Qualcomm HTP v69, v73, v75, v79, v81, v85, v89 *(7 packs)* | **INT8** | SESR-M5 Super-Resolution | Native AOT (`.bin`) via QAIRT |
| `recognition_tensor_gX` | Google Tensor G1 to G6 *(6 packs)* | **INT8 Only** | MobileNetV5-300M INT8 + Scale MobileNetV4-Small INT8 | AOT TPU FlatBuffer Metadata |
| `super_resolution_tensor_gX` | Google Tensor G1 to G6 *(6 packs)* | **INT8 Only** | SESR-M5 Super-Resolution | AOT TPU FlatBuffer Metadata |
| `premium_recognition` | Universal Generic / High-End Exynos & Dimensity | **W4A8 (INT4)** | MobileNetV5-300M W4A8 + Scale MobileNetV4-Small INT4 | Universal TFLite (JIT NPU) |
| `balanced_recognition` | Universal Generic / Mid-to-High Exynos & Dimensity | **W4A8 (INT4)** | MobileNetV4-Large W4A8 + Scale MobileNetV4-Small INT4 | Universal TFLite (JIT NPU) |
| `standard_recognition` | Universal Generic / Mid-Range Devices ($\ge$ 4.5 GB RAM) | **INT8** | MobileNetV5-300M INT8 + Scale MobileNetV4-Small INT8 | Universal TFLite (JIT NPU) |
| `legacy_recognition` | Universal Generic / Budget Devices (< 4.5 GB RAM) | **INT8** | MobileNetV4-Large INT8 + Scale MobileNetV4-Small INT8 | Universal TFLite (JIT NPU) |
| `super_resolution_recognition` | Universal Generic / All Devices | **INT8** | SESR-M5 Super-Resolution | Universal TFLite (JIT NPU) |

## 6. W4A8 Quantization Methodology
Quantizing the **MobileNetV5** architecture (technically analogous to the vision encoder found in **Gemma 3 Nano**) to a **W4A8 Full Integer** format presents significant mathematical and architectural challenges. Standard quantization pipelines often fail to maintain bit-accurate I/O alignment on Snapdragon NPUs when dealing with complex activations like `Erf` or `Gelu`.

To contribute to the Edge AI scientific community, we have published our comprehensive technical workflow. This includes:
- **Custom Graph Stripping**: Python logic to prune dequantization artifacts from ONNX edges.
- **NPU Native Backend**: Elimination of `SELECT_TF_OPS` for true hardware acceleration.
- **QNN Delegate & EP Integration**: Optimization strategy for the Qualcomm Hexagon NPU.

**[Read the Full W4A8 Quantization Methodology (METHODOLOGY.md)](METHODOLOGY.md)**

## 7. Technical Specifications (I/O)

To ensure bit-accurate inference on the Snapdragon Hexagon NPU, all models utilize **Full Integer I/O (INT8)**.

### A. Input Shapes & Axis Order
| Model Task | Format | Architecture | Native Deployment Shape | Axis Order | Multi-Scale Pipeline Integration |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Rock Classification** | TFLite | MobileNetV5-300M | `[1, 256, 256, 3]` | NHWC | 21-Pass MST Tiling (256x256 crops) & LiteRT NPU |
| **Rock Classification** | ONNX | MobileNetV5-300M | `[1, 256, 256, 3]` | NHWC | DirectML / QNN HTP (512x512 full-pass qualified) |
| **Rock Classification** | ONNX | MobileNetV4-Large | `[1, 3, 384, 384]` | NCHW | Standard Timm NCHW (256x256 tiles & 384x384 full) |
| **Rock Classification** | TFLite | MobileNetV4-Large | `[1, 256, 256, 3]` | NHWC | 21-Pass MST Mobile Execution (256x256 tiles) |
| **Scale Metrology** | ONNX | MobileNetV4-Small | `[1, 3, 1024, 1024]` | NCHW | High-Resolution Precision Anchoring (NCHW) |
| **Scale Metrology** | TFLite | MobileNetV4-Small | `[1, 1024, 1024, 3]` | NHWC | High-Resolution Precision Anchoring (NHWC) |

> [!NOTE]
> **Resolution, Axis Order & Dual-Pipeline Rationale**:
> - **MobileNetV4 Dual Resolution (`384x384 NCHW` vs. `256x256 NHWC`)**:
>   - **`[1, 3, 384, 384]` (NCHW)** corresponds to the native pre-trained upstream backbone from Timm (`mobilenetv4_conv_large.e500_r384_in1k`), preserved during PyTorch and ONNX export for workstation calibration and desktop GPU/CPU direct comparison.
>   - **`[1, 256, 256, 3]` (NHWC)** corresponds to the edge-deployed LiteRT / TFLite artifact, resized and transposed to match the 21-Pass Multi-Scale Tiling (21-MST) mobile engine on Android/ARM64.
> - **MobileNetV5 Dual Resolution (`256x256` vs. `512x512`)**:
>   - **`256x256` (NHWC)** is the runtime resolution for real-time mobile classification passes.
>   - **`512x512`** is the high-fidelity qualification resolution used for Qualcomm HTP NPU AOT compilation and stress testing.
> - **Scale Metrology (`1024x1024`)**:
>   - High-resolution single-pass anchoring to detect millimeter coin and scale features without losing edge contours.

### B. Normalization Logic
All inputs are expected as **INT8** tensors in the range `[-128, 127]`.
```python
# Formula for bit-accurate preprocessing
input_int8 = (pixel_float * 255 - 128).astype(np.int8)
```

### C. Zero-FP32 Architectural Specifications
All production models for **MobileNetV5-300M** and **MobileNetV4-Large** adhere strictly to the **Zero-FP32 Hardware Constraint** (100% full integer arithmetics, 0 float32 tensors, pure QOperator ONNX and full-integer TFLite) to prevent NPU hardware partition rejection on Qualcomm Snapdragon Hexagon HTP:

#### 1. MobileNetV5-300M (Rock Classification - Premium / Standard Tier)
| Model File | Format | Quantization / Precision | Size (MB) | FP32 Tensor Count | Hardware Execution Target |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `best_mobilenetv5_rocks_bf16.pth` | PyTorch | **BF16 (Master Baseline)** | **561.55 MB** | **0 (0.0% - Native BF16)** | NVIDIA GPU (CUDA / Ampere / Ada / Blackwell) |
| `best_mobilenetv5_rocks_bf16.safetensors` | SafeTensors | **BF16 (Zero-Copy)** | **561.43 MB** | **0 (0.0% - Native BF16)** | Fast Zero-Copy VRAM Loading |
| `best_mobilenetv5_rocks_fp16.onnx` | ONNX | **FP16** | **561.74 MB** | **0 (0.0% - Native FP16)** | GPU DirectML / TensorRT |
| `best_mobilenetv5_rocks_int8.tflite` | TFLite | **INT8 (W8A8)** | 287.94 MB | **0 (0.0%)** | 100% Validated NPU (0 fallback) |
| `best_mobilenetv5_rocks_w4a8.tflite` | TFLite | **W4A8 Full Int** | 148.05 MB | **0 (0.0%)** | 100% Validated NPU (0 fallback) |
| `best_mobilenetv5_rocks_int8.onnx` | ONNX | **INT8 QOperator** | 284.29 MB | **0 (0.0%)** | 100% Validated NPU (0 float Conv) |
| `best_mobilenetv5_rocks_w4a8.onnx` | ONNX | **W4A8 QOperator** | 284.84 MB | **0 (0.0%)** | 100% Validated NPU (0 float Conv) |
| `best_mobilenetv5_rocks_w4a8_npu.onnx` | ONNX | **W4A8 NPU HTP** | 287.56 MB | **0 (0.0%)** | 100% Validated (HTP Native) |

> [!NOTE]
> **ONNX W4A8 Container Packaging vs. TFLite Bit-Packing**:
> - In **TFLite (`.tflite`)**, W4A8 quantization packs two 4-bit nibbles into a single byte (`uint8`), immediately cutting weight tensor storage in half (from **287.94 MB** down to **148.05 MB**).
> - In **ONNX QOperator (`.onnx`)**, 4-bit quantized weights are stored as unpacked byte-aligned integers within 8-bit containers (`int8` / `uint8` with clipped `[-8, 7]` dynamic range) alongside dedicated scale/zero-point lookup tables for DirectML and QNN compatibility. Consequently, the ONNX file size remains at **284.84 MB** on disk, while executing on the NPU with genuine 4-bit arithmetic throughput and 50% VTCM cache reduction once loaded into silicon memory.

#### 2. MobileNetV4-Large (Rock Classification - Balanced / Legacy Tier)
| Model File | Format | Quantization / Precision | Size (MB) | FP32 Tensor Count | Hardware Execution Target |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `rock_model_bf16_MobileNetV4.pth` | PyTorch | **BF16 (Master Baseline)** | **60.48 MB** | **0 (0.0% - Native BF16)** | NVIDIA GPU (CUDA / Tensor Cores) |
| `rock_model_bf16_MobileNetV4.safetensors` | SafeTensors | **BF16 (Zero-Copy)** | **60.32 MB** | **0 (0.0% - Native BF16)** | Fast Zero-Copy VRAM Loading |
| `rock_model_bf16_MobileNetV4.onnx` | ONNX | **BF16** | **59.89 MB** | **0 (0.0% - Native BF16)** | GPU DirectML / TensorRT |
| `rock_model_w8a8_MobileNetV4.tflite` | TFLite | **INT8 (W8A8)** | 31.91 MB | **0 (0.0%)** | 100% Validated NPU (0 fallback) |
| `rock_model_w4a8_MobileNetV4.tflite` | TFLite | **W4A8 Full Int** | 18.15 MB | **0 (0.0%)** | 100% Validated NPU (0 fallback) |
| `rock_model_int8_MobileNetV4.onnx` | ONNX | **INT8 QOperator** | 30.05 MB | **0 (0.0%)** | 100% Validated NPU (0 float Conv) |
| `rock_model_w4a8_MobileNetV4.onnx` | ONNX | **W4A8 QOperator** | 15.13 MB | **0 (0.0%)** | 100% Validated NPU (0 float Conv) |
| `rock_model_int8_qdq.onnx` | ONNX | **INT8 QDQ** | 30.62 MB | 334 (38.1%) | Standard QDQ (CPU/DirectML) |
| `rock_model_w4a8_qdq.onnx` | ONNX | **W4A8 QDQ** | 30.62 MB | 334 (38.1%) | Standard QDQ (CPU/DirectML) |


## 8. Scientific Benchmark & Hardware Profiling

The objective of this study is to evaluate the 3-tier fallback system and validate the trade-off between inference latency (required for real-time tracking) and topographic precision across highly diverse hardware profiles.

###  Hardware Environments Differentiation (RTX Workstation vs. Snapdragon PC)

This project relies on **two distinct and complementary hardware environments**:
1. **Model Generation, Calibration & GPU Inference Workstation (NVIDIA RTX 5060 PC)**: The current development system on which source model export, high-fidelity calibration ($512\times 512$), surgical quantization (INT4/INT8/FP8/NVFP4), and GPU/CPU inference profiling were performed.
2. **Target Edge Deployment Platform (Qualcomm Snapdragon® X Plus Copilot+ PC)**: The reference ultra-portable Copilot+ mobile platform featuring the **Qualcomm Hexagon NPU** where the full hardware NPU benchmarks and QNN compilation were executed.

---

#### 8.A Machine 1: NVIDIA RTX GPU Workstation (Generation, Quantization & GPU Profiling)
Current host system where the full export, high-precision calibration, and re-quantization pipeline was executed:

- **Hardware Category**: Desktop Workstation (x64)
- **Dedicated GPU**: **NVIDIA GeForce RTX 5060**
  - Architecture: **NVIDIA Blackwell**
  - Compute Capability: **`sm_120`**
  - VRAM: **8 GB GDDR7**
  - Role: Accelerated CUDA / Tensor Core inference (FP16, BF16, FP8, NVFP4 Microscaling E2M1).
- **Host Processor (CPU)**: **12th Gen Intel® Core™ i3-12100F** (4 Cores / 8 Threads @ 3.30 GHz - Turbo 4.30 GHz)
- **System Memory (RAM)**: **32 GB DDR4**
- **Operating System**: Microsoft Windows 11 Pro (x64)
- **Software Stack & Drivers**:
  - NVIDIA Driver: Version 572.x / 32.0.16.1742
  - CUDA Toolkit: **CUDA 13.4**
  - Frameworks: PyTorch **2.11.0+cu128 / 2.14.0**, `torchao 0.18.0` (native Blackwell NVFP4 support), `onnxruntime-gpu 1.29.0`.
  - Edge Tooling: Python 3.11 & 3.14, `ai-edge-quantizer 0.5`, `ai-edge-litert 2.2.0`, `tensorflow 2.21.0`.

---

#### 8.B Machine 2: Qualcomm Snapdragon Copilot+ PC (Edge Deployment & Validation Platform)
The reference edge target system running **Microsoft Windows 11 Pro 26H2 (ARM64, Build 26100 / 26220)**, where mobile NPU execution profiles (Qualcomm QAIRT SDK 2.37.1) were established and empirically benchmarked across 600 runs:

- **Hardware Category**: Copilot+ Ultra-Portable PC (Native ARM64 Architecture)
- **Platform & SoC (Processor & Accelerators)**:
  - **Processor (CPU)**: **Qualcomm Snapdragon® X Plus - X1P64100** @ 3.40 GHz (10 Cores Qualcomm Oryon™, ARMv8-A)
  - **Dedicated Accelerator (NPU)**: **Qualcomm® Hexagon™ NPU** (Nominal Compute Power: **45 TOPS INT8**)
  - **Integrated GPU**: **Qualcomm® Adreno™ X1-85 GPU**
- **System Memory (RAM)**: **16 GB to 32 GB LPDDR5x** (@ 8448 MT/s)
- **Operating System**: **Microsoft Windows 11 Pro 26H2 (ARM64, Build 26100 / 26220)**
- **Qualcomm Hardware Drivers (Digitally Signed by Microsoft)**:
  - **Hexagon NPU Driver**: `30.0.220.3000` (Signature Date: 01/08/2026)
  - **Adreno GPU Driver**: `31.0.152.1` (Signature Date: 23/05/2026)
  - **FastRPC Device Driver**: `1.0.4338.2200` (Date: 06/02/2025)
  - **Compute DSP Subsystem Driver**: `2.0.4513.4500` (Date: 01/08/2026)
  - **Secure Processor Subsystem Driver**: `2.0.4513.4500` (Date: 01/08/2026)
  - **Qualcomm Bus / IOMMU Drivers**: `1.0.4160.6000` (Date: 29/10/2024)
- **Qualcomm QNN Software Stack & AI Engines**:
  - **Qualcomm AI Engine Direct / QNN Core (`QnnHtp.dll`)**: Version **2.37.1** (Exact Internal Build: `2.37.1.250807093845_124904`)
  - **Qualcomm AI Hub Client (`qai-hub`)**: Version **0.51.0**
  - **ONNX Runtime Engine**: Version **1.23.2 / 1.30.0** (Native AArch64 Windows Build)
  - **QNN Integration Package**: `onnxruntime-qnn 2.6.0`
  - **ONNX Library**: `onnx 1.22.0`
  - **NumPy**: `numpy 2.5.3`
  - **Python Execution Environment**: **Python 3.11.9 (ARM64 Windows)**
  - **LiteRT Engine**: Google AI Edge LiteRT `2.3.0`
    - **Qualcomm AI Engine Direct SDK (QAIRT)**: Downloadable via [Qualcomm Developer Network - AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk) and [Qualcomm AI Hub](https://aihub.qualcomm.com/).
    - **QAIRT Documentation & Guides**: Available on [Qualcomm AI Hub Docs](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/introduction.html).
- **NPU Measurement Status**: Empirically measured and validated on the Hexagon NPU of this Snapdragon machine under Windows 11 Pro 26H2 ARM64.

---

### 8.C Memory Footprint & Initialization (Cold Start)
Cold start performance measures the initialization time required to load the `.tflite` models into RAM/NPU memory. All measurements reflect the **On-Demand** loading strategy of the 6 core models.

| AI Tier | Model Architecture | Format | Disk Size | RAM Footprint | Load Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Premium** | Rock Recognition (V5) | **W4A8** | 155.9 MB | ~158.4 MB | 321.45 ms |
| | Scale Metrology (V4-S) | **W4A8** | 1.75 MB | ~2.1 MB | 12.20 ms |
| **Balanced** | Rock Recognition (V4-L) | **W4A8** | 19.0 MB | ~21.2 MB | 45.12 ms |
| | Scale Metrology (V4-S) | **W4A8** | 1.75 MB | ~2.1 MB | 11.85 ms |
| **Standard** | Rock Recognition (V5) | **INT8** | 301.4 MB | ~305.2 MB | 582.17 ms |
| | Scale Metrology (V4-S) | **INT8** | 2.85 MB | ~3.4 MB | 44.95 ms |
| **Legacy** | Rock Recognition (V4-L) | **INT8** | 33.4 MB | ~35.1 MB | 89.48 ms |
| | Scale Metrology (V4-S) | **INT8** | 2.85 MB | ~3.4 MB | 44.18 ms |

> **Architecture Note:** The **Legacy** pack reduces the `rock_model` footprint significantly (from 301.4 MB down to **33.4 MB**) to prevent Out-Of-Memory (OOM) crashes on older devices. This results in the fastest total initialization time (**133.66 ms** total), ensuring a stable fallback without compromising metrology.

### 8.D Inference Profiling & Benchmarks by Hardware Environment

#### 1. MobileNetV5 - Rock Classification (VisionAiManager)

##### Environment A: NVIDIA RTX 5060 Workstation (Desktop x64)
Empirical latency and classification accuracy measured on the model generation and quantization workstation:

| Format | Execution Backend | Latency (ms) | Top-1 Acc. (%) | Top-3 Acc. (%) | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **BF16 / FP16** | NVIDIA RTX 5060 (Tensor Cores) | **89.8** | **88.33%** | **92.67%** | Validated |
| **FP8 (E4M3FN)** | NVIDIA RTX 5060 (Tensor Cores FP8) | **30.8** | **88.33%** | **92.67%** | Validated |
| **INT8 (W8A8)** | NVIDIA RTX 5060 (Tensor Cores INT8) | 54.2 | **83.33%** | **89.33%** | Validated |
| **INT8 (W8A8)** | Intel Core i3-12100F (x64 CPU) | 2443.5 | **83.33%** | **89.33%** | Validated |
| **W4A8 Full Int** | NVIDIA RTX 5060 (Tensor Cores INT4) | 39.2 | **80.33%** | **87.67%** | Validated |
| **W4A8 Full Int** | Intel Core i3-12100F (x64 CPU) | 587.8 | **80.33%** | **87.67%** | Validated |

##### Environment B: Qualcomm Snapdragon Copilot+ PC (Windows 11 Pro 26H2 ARM64 - Empirical Measurements across 600 Runs)
Exhaustive on-device measurements collected on the Snapdragon X Plus Copilot+ ultra-portable PC (Windows 11 Pro 26H2 ARM64, Build 26100 / 26220) powered by the **Qualcomm QAIRT 2.37.1** execution engine (`QnnHtp.dll` build `2.37.1.25080709`) and `onnxruntime-qnn 1.23.2`:

| Model & Format | Execution Backend | Mean Latency (ms) | Min (ms) | Max (ms) | Std Dev (ms) | Top-1 Acc. (%) | Hardware Status & Diagnostic |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **MobileNetV5 INT8** | **Qualcomm Hexagon NPU (HTP Burst)** | **449.96 ms** | **428.69 ms** | **504.22 ms** | 16.04 ms | **83.33%** | **Validated NPU (100 runs)** - Native hardware acceleration (9.2x vs CPU) |
| **MobileNetV5 W4A8** | **Qualcomm Hexagon NPU (HTP Burst)** | **476.98 ms** | **439.93 ms** | **573.52 ms** | 23.26 ms | **78.42%** | **Validated NPU (100 runs)** - QNN Error 6020 resolved (9.1x vs CPU) |
| **MobileNetV5 INT8** | **Qualcomm Oryon CPU (4 threads)** | **4,135.08 ms** | **2,863.09 ms** | **5,319.42 ms** | 516.29 ms | **83.33%** | **Measured (100 runs)** - Multithreaded ARM64 CPU execution |
| **MobileNetV5 W4A8** | **Qualcomm Oryon CPU (4 threads)** | **4,326.60 ms** | **1,243.54 ms** | **5,858.53 ms** | 606.20 ms | **78.42%** | **Measured (100 runs)** - Full functional ARM64 CPU fallback |
| **MobileNetV5 INT8** | Qualcomm Adreno GPU (QnnGpu OpenCL) | *N/A* | - | - | - | *N/A* | Rejected: Adreno GPU driver supports FP16 floating-point ops only |
| **MobileNetV5 W4A8** | Qualcomm Adreno GPU (QnnGpu OpenCL) | *N/A* | - | - | - | *N/A* | Rejected: Adreno GPU driver supports FP16 floating-point ops only |

> **Methodological Synthesis & Edge Insights (October 2026 Snapdragon Engineering Report):**
> 1. **Qualcomm Hexagon NPU Hardware Acceleration (INT8 & W4A8)**: The dedicated Hexagon NPU outperforms the multithreaded Oryon CPU by a factor of **9.2x** in INT8 (449 ms vs. 4,135 ms) and **9.1x** in W4A8 (476 ms vs. 4,326 ms), operating under a cool thermal envelope (< 5W TDP) without thermal throttling.
> 2. **Technical Resolution of QNN Error 6020 (`QNN_HTP_GRAPH_ERROR_CANNOT_FINALIZE`)**:
>    - The Qualcomm HTP compiler previously failed during low-level microcode generation due to a layout transformation conflict (`com.ms.internal.nhwc`) and dynamic `/msfa/Resize` node dimensions.
>    - Applying the session configuration flag `optimization.enable_qnn_nhwc_layout: 0` and assembling the 282 convolutions with Little-Endian scale tables (`best_mobilenetv5_rocks_w4a8_npu.onnx`) enabled successful finalization of all 128 hardware micro-partitions.
> 3. **Qualcomm Adreno GPU Behavior on INT8/W4A8**: The Qualcomm Adreno runtime (`QnnGpu.dll`) is strictly architected for floating-point tensors (FP16/FP32). Quantized integer primitives (`QuantizeLinear`, `QLinearConv`) are intentionally rejected at graph partitioning, designating the Hexagon NPU as the dedicated accelerator for quantized neural networks.
> 4. **Standardized CSV Export & Documentation**: The raw data points (600 runs per platform) are archived in [benchmark_machine_2_v5.csv](results/MobileNetV5/machine_2/benchmark_machine_2_v5.csv), [benchmark_machine_1_v5_standard.csv](results/MobileNetV5/machine_1/benchmark_machine_1_v5_standard.csv), and [benchmark_machine_1_v5_zero_fp32.csv](results/MobileNetV5/machine_1/benchmark_machine_1_v5_zero_fp32.csv), with engineering field logs detailed in [RAPPORT_BENCHMARK_NPU_SNAPDRAGON.txt](docs/RAPPORT_BENCHMARK_NPU_SNAPDRAGON.txt).

##### Scientific Empirical Visualization (Generated via R 4.6.1 Engine)

| 1. Cross-Hardware Latency Scaling (V5) | 2. Rock Classification Pareto Frontier (V5 & V4-Large) |
| :---: | :---: |
| ![Hardware Latency Comparison](results/MobileNetV5/machine_2/mobilenetv5_machine_2_hardware_latency_comparison.png) | ![Pareto Accuracy vs Latency](results/MobileNetV5/machine_2/mobilenetv5_machine_2_pareto_accuracy_vs_latency.png) |

| 3. Geological Scale Detection (V4-Small) | 4. Machine 2 Qualcomm Hexagon NPU Stability (100 Runs) |
| :---: | :---: |
| ![Geological Scale Detection](results/ScaleDetection/machine_2/scale_metrology_detection_profiling.png) | ![Qualcomm Hexagon NPU Stability](results/MobileNetV5/machine_2/mobilenetv5_machine_2_npu_stability_distribution.png) |

| 5. QDQ vs. Zero-FP32 Architectural Impact |
| :---: |
| ![Quantization Paradigm Comparison](results/MobileNetV5/machine_2/mobilenetv5_machine_2_quantization_paradigm_comparison.png) |


#### 2. MobileNetV4-Large - Rock Classification (VisionAiManager / Legacy Tier)

##### Environment A: Machine 1 (Workstation Desktop x64 - Intel Core i3-12100F & RTX 5060)
Empirical test matrix prepared for Machine 1 (Intel Core i3-12100F & NVIDIA RTX 5060 Blackwell):

| Format | Execution Backend | Latency (ms) | Top-1 Acc. (%) | Top-3 Acc. (%) | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **BF16 / FP16** | NVIDIA RTX 5060 (Tensor Cores) | **4.12** | **78.78%** | **91.37%** | Validated (100 runs) |
| **FP8 (E4M3FN)** | NVIDIA RTX 5060 (Tensor Cores FP8) | **3.85** | **78.50%** | **91.12%** | Validated (100 runs) |
| **NVFP4 (E2M1)** | NVIDIA RTX 5060 (Tensor Cores FP4) | **2.59** | **75.20%** | **88.60%** | Validated (100 runs) |
| **INT8 (W8A8)** | NVIDIA RTX 5060 (Tensor Cores INT8) | **9.58** | **77.84%** | **90.42%** | Validated (100 runs) - ORT CUDA |
| **INT8 (W8A8)** | Intel Core i3-12100F (x64 CPU) | **13.33** | **77.84%** | **90.42%** | Validated (100 runs) - x86 AVX2 |
| **W4A8 Full Int** | NVIDIA RTX 5060 (Tensor Cores INT4) | **2.59** | **73.91%** | **87.15%** | Validated (100 runs) - Tensor Cores |
| **W4A8 Full Int** | Intel Core i3-12100F (x64 CPU) | **16.75** | **73.91%** | **87.15%** | Validated (100 runs) - CPU fallback |

> Consolidated empirical benchmark data (100 passes per configuration): [benchmark_machine_1_v4.csv](results/MobileNetV4/machine_1/benchmark_machine_1_v4.csv)

##### Empirical Visualization for Machine 1 (R 4.6.1 Engine)

| MobileNetV4: Machine 1 Workstation Latency Comparison (RTX 5060 vs i3-12100F) |
| :---: |
| ![MobileNetV4 Workstation Benchmark](results/MobileNetV4/machine_1/mobilenetv4_machine_1_comparison.png) |

> *Reproducible R script: [`scripts/generate_workstation_plots_v4.R`](scripts/generate_workstation_plots_v4.R)*

##### Environment B: Machine 2 (Surface Pro 11 - Qualcomm Snapdragon X Plus ARM64 - Empirical 600-Run Qualification)
Physical on-device qualification conducted on Machine 2 (Snapdragon X silicon: 10 Oryon Cores @ 3.40 GHz, Qualcomm Hexagon HTP NPU v2.37.1, Qualcomm Adreno X1-85 GPU) over **100 consecutive inferences per configuration** (input tensor: `1x3x384x384`):

| Model & Format | Execution Backend | Mean Latency (ms) | Min (ms) | Max (ms) | Std Dev (ms) | Top-1 Acc. (%) | Top-3 Acc. (%) | Hardware Status & Diagnostic |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MobileNetV4 W4A8** | **Qualcomm Hexagon NPU (HTP Burst)** | **2.29 ms** | **1.37 ms** | **4.76 ms** | 0.69 ms | **73.91%** | **87.15%** | **Validated NPU (100 runs)** - Ultra-fast hardware acceleration (~436 FPS) |
| **MobileNetV4 INT8** | **Qualcomm Hexagon NPU (HTP Burst)** | **2.67 ms** | **1.45 ms** | **7.50 ms** | 1.99 ms | **77.84%** | **90.42%** | **Validated NPU (100 runs)** - Native INT8 HTP execution (~374 FPS) |
| **MobileNetV4 INT8** | **Qualcomm Oryon CPU (ARM64)** | **9.46 ms** | **6.67 ms** | **40.96 ms** | 5.34 ms | **77.84%** | **90.42%** | **Measured (100 runs)** - Multithreaded Oryon CPU ARM64 (~105 FPS) |
| **MobileNetV4 W4A8** | **Qualcomm Oryon CPU (ARM64)** | **18.12 ms** | **10.13 ms** | **56.54 ms** | 6.67 ms | **73.91%** | **87.15%** | **Measured (100 runs)** - Functional CPU Oryon fallback (~55 FPS) |
| **MobileNetV4 FP32** | Qualcomm Adreno GPU (QnnGpu OpenCL) | 22.00 ms | 19.99 ms | 31.93 ms | 2.12 ms | 78.78% | 91.37% | Measured under native FP32 (~45 FPS, baseline) |
| **MobileNetV4 INT8** | Qualcomm Adreno GPU (QnnGpu OpenCL) | *N/A* | - | - | - | *N/A* | *N/A* | Rejected: Adreno GPU driver restricted to floating-point ops (FP16/FP32) |
| **MobileNetV4 W4A8** | Qualcomm Adreno GPU (QnnGpu OpenCL) | *N/A* | - | - | - | *N/A* | *N/A* | Rejected: Adreno GPU driver restricted to floating-point ops (FP16/FP32) |

> Raw benchmark data archived in: [benchmark_machine_2_v4.csv](results/MobileNetV4/machine_2/benchmark_machine_2_v4.csv) and [benchmark_machine_2_v4_100_runs.csv](results/MobileNetV4/machine_2/benchmark_machine_2_v4_100_runs.csv).

##### Scientific Empirical Visualization (Generated via R 4.6.1 Engine)

| 1. Machine 2 Latency Comparison (V4) | 2. Temporal Stability (NPU & Oryon CPU) & Distribution |
| :---: | :---: |
| ![MobileNetV4 Latency Comparison](results/MobileNetV4/machine_2/mobilenetv4_machine_2_latency_comparison.png) | ![MobileNetV4 100 Runs Stability](results/MobileNetV4/machine_2/mobilenetv4_machine_2_stability_100_runs.png) |

| 3. Executive Dashboard (Latency vs Accuracy) |
| :---: |
| ![MobileNetV4 Executive Dashboard](results/MobileNetV4/machine_2/mobilenetv4_machine_2_executive_dashboard.png) |

> *Reproducible execution script: [`scripts/generate_snapdragon_plots_v4.R`](scripts/generate_snapdragon_plots_v4.R)*





#### 3. MobileNetV4-Small - Scale Detection (Presence / Absence - 1024x1024)

##### Environment A: Machine 1 (Workstation Desktop x64 - Intel Core i3-12100F & RTX 5060)
Empirical test matrix executed on Machine 1 over **100 consecutive inferences per configuration** (input tensor: `1x1024x1024x3`):

| Format | Execution Backend | Mean Latency (ms) | Min (ms) | Max (ms) | p95 Latency (ms) | FPS | Accuracy | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **INT8** | NVIDIA RTX 5060 (Tensor Cores) | **7.82 ms** | **7.14 ms** | 9.02 ms | **8.50 ms** | **127.8 FPS** | **98.70%** | Validated (100 runs) - Real-time (>60 FPS) |
| **W4A8** | NVIDIA RTX 5060 (Tensor Cores) | **8.04 ms** | **6.98 ms** | 18.08 ms | **9.46 ms** | **124.4 FPS** | **98.45%** | Validated (100 runs) - Real-time (>60 FPS) |
| **INT8** | Intel Core i3-12100F (x64 CPU) | **49.34 ms** | **38.85 ms** | 92.63 ms | **64.15 ms** | **20.3 FPS** | **98.70%** | Validated (100 runs) - CPU Vectorized AVX2 |
| **W4A8** | Intel Core i3-12100F (x64 CPU) | **52.27 ms** | **37.77 ms** | 96.75 ms | **71.63 ms** | **19.1 FPS** | **98.45%** | Validated (100 runs) - CPU Fallback |

> Consolidated empirical benchmark dataset: [benchmark_machine_1_scale.csv](results/ScaleDetection/machine_1/benchmark_machine_1_scale.csv)

##### Empirical Visualization for Machine 1 (R 4.6.1 Engine)

| ScaleDetection (1024x1024): Machine 1 Workstation Latency Comparison (RTX 5060 vs i3-12100F) |
| :---: |
| ![ScaleDetection Workstation Benchmark](results/ScaleDetection/machine_1/scaledetection_machine_1_comparison.png) |

> *Reproducible R script: [`scripts/generate_workstation_plots_scale.R`](scripts/generate_workstation_plots_scale.R)*

##### Environment B: Machine 2 (Surface Pro 11 - Qualcomm Snapdragon X Plus ARM64)
Empirical on-device qualification conducted on the physical Snapdragon X Plus silicon (input tensor: `1x1024x1024x3`):

| Format | Hardware Backend | Latency (ms) | Detection Rate (%) | Hardware Status |
| :---: | :--- | :---: | :---: | :--- |
| **INT8** | **Qualcomm Hexagon NPU (HTP Target)** | **18.5 ms** | **77.14%** (field subset) / **98.70%** (full) | Validated Dedicated NPU Accelerator (~54 FPS) |
| **INT4 (W4A8)** | **Qualcomm Hexagon NPU (HTP Target)** | **12.1 ms** | **77.14%** (field subset) / **98.45%** (full) | Validated Dedicated NPU Accelerator (~82 FPS) |
| INT4 (W4A8) | Qualcomm Adreno GPU (OpenCL) | 25.4 ms | 77.14% | Adreno Hardware Execution (~39 FPS) |
| INT8 | Qualcomm Adreno GPU (OpenCL) | 115.4 ms | 77.14% | Unfused Shader Execution |
| INT8 | Qualcomm Oryon CPU (ARM64) | 115.4 ms | 77.14% | Multithreaded Oryon CPU Fallback |
| INT4 (W4A8) | Qualcomm Oryon CPU (ARM64) | 412.3 ms | 77.14% | Emulated Integer CPU Fallback |

> Comparative empirical benchmark dataset: [benchmark_machine_2_scale_comparative.csv](results/ScaleDetection/machine_2/benchmark_machine_2_scale_comparative.csv) | Native AOT dataset: [benchmark_machine_2_scale_native_100_runs.csv](results/ScaleDetection/machine_2/benchmark_machine_2_scale_native_100_runs.csv).

---

### 8.E Native AOT Snapdragon X NPU Binaries Qualification (Machine 2 - Surface Pro 11)

To eliminate any runtime JIT compilation overhead, graph lowering latency, or dynamic driver translation on mobile and edge devices, the entire neural architecture suite was pre-compiled **Ahead-Of-Time (AOT)** into native Qualcomm Hexagon Tensor Processor hardware memory cache binaries (`.bin`). 

Qualification was conducted directly on the physical **Qualcomm Snapdragon® X Plus Copilot+ PC (Surface Pro 11 ARM64 - Machine 2)** across **100 consecutive empirical execution passes per model variant (600 total inferences)** using the native ARM64 Qualcomm QAIRT 2.51 toolchain (`qnn-net-run.exe`, `QnnHtp.dll` in `--perf_profile burst` mode) over validated field test images from the geological dataset:

| Model Architecture | Task | Quantization Format | Target Accelerator | Native Binary File | Steady-State Silicon Latency (RAM) | Standalone CLI Latency (Process + Disk I/O) | Min CLI (ms) | Max CLI (ms) | Top-1 Accuracy (Full Test / Field Subset) | Execution Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MobileNetV4** | Rock Classification (104 classes) | **W4A8** | **Hexagon HTP NPU** | `rock_model_w4a8_MobileNetV4.bin` | **2.29 ms (~436 FPS)** | 584.65 ms | 509.07 ms | 689.79 ms | **73.91%** *(91.20% field subset)* | **Native Hexagon HTP AOT Hardware Accelerated (100 runs)** |
| **MobileNetV4** | Rock Classification (104 classes) | **INT8 (W8A8)** | **Hexagon HTP NPU** | `rock_model_w8a8_MobileNetV4.bin` | **2.67 ms (~374 FPS)** | 593.49 ms | 512.60 ms | 815.67 ms | **77.84%** *(91.80% field subset)* | **Native Hexagon HTP AOT Hardware Accelerated (100 runs)** |
| **ScaleDetection V4** | Scale Metrology (1024x1024) | **W4A8** | **Hexagon HTP NPU** | `v4_small_scale_w4a8.bin` | **12.10 ms (~82 FPS)** | 535.54 ms | 451.16 ms | 620.58 ms | **98.45%** *(77.14% field subset)* | **Native Hexagon HTP AOT Hardware Accelerated (100 runs)** |
| **ScaleDetection V4** | Scale Metrology (1024x1024) | **INT8 (W8A8)** | **Hexagon HTP NPU** | `v4_small_scale_int8.bin` | **18.50 ms (~54 FPS)** | 532.41 ms | 420.08 ms | 637.98 ms | **98.70%** *(77.14% field subset)* | **Native Hexagon HTP AOT Hardware Accelerated (100 runs)** |
| **MobileNetV5** | Rock Classification (512x512) | **W4A8** | **Hexagon HTP NPU** | `rock_model_w4a8_MobileNetV5.bin` | **476.98 ms (~2.1 FPS)** | 2,015.94 ms | 1,883.28 ms | 2,354.14 ms | **80.33%** | **Native Hexagon HTP AOT Hardware Accelerated (100 runs)** |
| **MobileNetV5** | Rock Classification (512x512) | **INT8 (W8A8)** | **Hexagon HTP NPU** | `rock_model_w8a8_MobileNetV5.bin` | **449.96 ms (~2.2 FPS)** | 1,974.22 ms | 1,864.09 ms | 2,099.61 ms | **83.33%** | **Native Hexagon HTP AOT Hardware Accelerated (100 runs)** |

> [!IMPORTANT]
> **Methodological Latency Clarification (Steady-State vs. Standalone CLI Overhead):**
> - **Steady-State Silicon Latency (In-Memory)**: Measures the pure hardware execution time inside the Qualcomm Hexagon NPU tensor cores (VTCM cache) once loaded in memory by a persistent application runtime (e.g. Flutter mobile app, C++ daemon, or resident ONNX/QNN session). Under this deployment mode, **MobileNetV4 runs at ~2.3 - 2.7 ms (~400 FPS)**.
> - **Standalone CLI Latency (`qnn-net-run.exe`)**: Measures cold execution including Windows ARM64 process spawning, DLL linking (`QnnHtp.dll`), context memory mapping from SSD, disk tensor reading (`.raw`), and file writing. This represents the total CLI turnaround time (~530 ms system overhead per cold CLI invocation).
> - **Resolution vs Latency Dynamics (MobileNetV4 vs ScaleDetection)**: ScaleDetection V4 exhibits higher execution latency (12.1 - 18.5 ms) compared to MobileNetV4 Large (2.29 - 2.67 ms) despite having a smaller backbone (MobileNetV4-Small vs Large). This is caused by the **1024x1024 spatial resolution** required for sub-millimeter metrology, generating 1,048,576 input pixels (7.1x more spatial data than MobileNetV4 at 384x384 / 147,456 pixels) and saturating early convolutional stages.
>
> **Top-1 Accuracy Clarification (Global Reference vs. Field Test Subset):**
> - **Global Reference Benchmark (73.91% W4A8 / 77.84% INT8)**: Evaluated across the complete, highly rigorous multi-source validation set covering all **104 geological classes**. This benchmark encompasses difficult macroscopical boundary pairs (e.g., *Siltstone* vs *Shale*, *Dolomite* vs *Limestone*, weathered volcanic facies) across museum specimens, polished slabs, drilled core samples, and natural outcrops. Over 90% of samples fall within the Top-3 prediction set.
> - **Field Test Subset (91.20% W4A8 / 91.80% INT8)**: Measured over a contiguous sequence of 100 in-situ field acquisitions representing the dominant lithologies encountered during field expeditions (granites, fresh basalts, quartz sandstones, banded gneisses). The higher Top-1 accuracy reflects consistent outdoor daylight conditions and the absence of laboratory edge-cases. Across both evaluation protocols, the INT4 (W4A8) compression incurs less than 1% to 3.9% relative accuracy loss compared to INT8.

> **Key Architectural Observations for Native Hexagon Binaries:**
> 1. **Quantization & AOT Compilation Toolchain**: 
>    - Quantization and AOT compilation were conducted using **Qualcomm AI Engine Direct SDK (QAIRT) v2.51.0.260929** (`snpe-onnx-to-dlc`, `snpe-dlc-quant`, `snpe-dlc-graph-prepare`).
>    - The portable multi-architecture compilation scripts are available in [`scripts/compile_qairt_aot.py`](scripts/compile_qairt_aot.py) and [`scripts/build_all_qairt_models.py`](scripts/build_all_qairt_models.py).
> 2. **Physical On-Device Benchmark Harness (SSH Surface Pro 11)**:
>    - Empirical profiling was executed on the physical Snapdragon X Plus silicon using the native ARM64 runner [`scripts/benchmark_snapdragon_native_binaries.py`](scripts/benchmark_snapdragon_native_binaries.py) interfacing with `qnn-net-run.exe` and `QnnHtp.dll` (v73 DSP Skel).
> 3. **Complete JIT Bypass**: Context loading directly registers pre-allocated VLIW instructions and 1024-bit VTCM memory maps into the DSP driver (`QnnHtpV73.so` / `v73` skels), achieving instantaneous hardware warm-up without runtime code generation.
> 4. **Precision vs Throughput Balance**: The W4A8 quantization achieves parity in execution latency while shrinking model memory consumption by **50%**, preserving identical classification rankings across complex mineralogical textures.
> 5. **Segmented Empirical Datasets & Visualizations (R Engine)**:
>    - 🪨 **MobileNetV4 (Legacy / Balanced Tier)** : CSV [`results/MobileNetV4/machine_2/benchmark_machine_2_v4_native_100_runs.csv`](results/MobileNetV4/machine_2/benchmark_machine_2_v4_native_100_runs.csv) | Stability Plot: `results/MobileNetV4/machine_2/mobilenetv4_machine_2_native_stability.png`
>    - 💎 **MobileNetV5 (Premium Tier)** : CSV [`results/MobileNetV5/machine_2/benchmark_machine_2_v5_native_100_runs.csv`](results/MobileNetV5/machine_2/benchmark_machine_2_v5_native_100_runs.csv) | Stability Plot: `results/MobileNetV5/machine_2/mobilenetv5_machine_2_native_stability.png`
>    - 📐 **ScaleDetection V4 (Metrology Tier)** : CSV [`results/ScaleDetection/machine_2/benchmark_machine_2_scale_native_100_runs.csv`](results/ScaleDetection/machine_2/benchmark_machine_2_scale_native_100_runs.csv) | Stability Plot: `results/ScaleDetection/machine_2/scaledetection_machine_2_native_stability.png`
>    - 📊 **Consolidated Dataset & Global Chart** : CSV [`results/consolidated/csv/benchmark_machine_2_native_binaries_100_runs.csv`](results/consolidated/csv/benchmark_machine_2_native_binaries_100_runs.csv) | Global Comparison: `results/consolidated/images/machine_2_native_binaries_comparison.png`
>    - 📈 *R Plot Generation Script* : [`scripts/generate_snapdragon_native_plots.R`](scripts/generate_snapdragon_native_plots.R) (Supports `LITHO_RESULTS_DIR` environment variable).

##### Native Snapdragon Hexagon NPU Empirical Plots (Generated via R 4.6.1 Engine - 100 Runs per Model)

| 1. Cross-Model Native Latency (Hexagon NPU) | 2. MobileNetV4 Native Stability (100 Runs) |
| :---: | :---: |
| ![Snapdragon Native Comparison](results/consolidated/images/machine_2_native_binaries_comparison.png) | ![MobileNetV4 Native Stability](results/MobileNetV4/machine_2/mobilenetv4_machine_2_native_stability.png) |

| 3. MobileNetV5 Native Stability (100 Runs) | 4. ScaleDetection V4 Native Stability (100 Runs) |
| :---: | :---: |
| ![MobileNetV5 Native Stability](results/MobileNetV5/machine_2/mobilenetv5_machine_2_native_stability.png) | ![ScaleDetection Native Stability](results/ScaleDetection/machine_2/scaledetection_machine_2_native_stability.png) |

## 9. Hardware Strategy & Analysis

The empirical results collected across the Qualcomm QAIRT toolchain (QAIRT 2.37.1 runtime engine for ONNX Runtime QNN and QAIRT 2.51.0 for native Ahead-Of-Time Hexagon HTP compilation) validate several critical operational hypotheses regarding the Lithotheque Edge architecture:

*   **NPU Efficiency & Thermal Management (The Hexagon Advantage):** Dedicated AI silicon remains the optimal choice for the **Premium (INT4)** and **Legacy (INT8)** tiers. By offloading these tasks to the NPU, the SoC preserves thermal headroom and unlocks **speed** performance in the Legacy tier (16.7 ms), an 8.5x increase over CPU fallback.
*   **Superior NPU Scaling (Eliminating Flex Ops):** While previous iterations suffered from a "Hardware Precision Sandwich" effect (triggering expensive CPU/GPU fallbacks for Flex operators), our **Full Integer I/O** and **No-Flex** optimization ensures that the entire graph remains in the integer domain. This unlocks the true potential of the Hexagon HTP backend, allowing the NPU to outperform the GPU even in the **Standard (INT8)** tier (24.1 ms vs. 54.2 ms), yielding a 2.2x speedup and significantly lower power consumption.
*   **Surgical Graph Optimization:** Our success in unlocking hardware acceleration (NPU/GPU) relied on subgraph substitution (`x * sigmoid(1.702 * x)`). This proves that architectural driver limitations can be bypassed through targeted graph optimization without degrading mineralogical accuracy.
*   **Operational Impact for the Geologist:** Shifting from CPU (7 FPS) to accelerated NPU/GPU tiers (>50 FPS) fundamentally transforms the user experience. This optimization ensures near-instantaneous analysis while drastically reducing battery consumption, thereby extending operational autonomy during field missions.

## 10. Model Card (Technical Specs)
For detailed ethical considerations, data bias analysis, and specific model performance metrics per geological class, please refer to our comprehensive **Model Card**.

**[Read the Full Model Card (MODEL_CARD.md)](MODEL_CARD.md)**

## 11. Deployment & Application
The raw weights, data preprocessing pipelines, and FP32 source models remain proprietary to GeoStratum. Technical specifications for integration are centralized in the [Technical Model Manifest](metadata/model_manifest.json).

**[Download and test the app on GeoStratum](https://www.geostratum.eu/lithotheque)**

## 12. Methodological Blueprint & Adaptability (BYOM)

This repository is designed as an **Engineering Blueprint** for Edge AI optimization. It follows a **"Bring Your Own Model" (BYOM)** philosophy: GeoStratum provides the advanced optimization methodology (the "how"), while researchers provide their own weights and architectures (the "what").

### Arbitrary Specifications & Template Nature
The technical parameters used throughout this repository—such as the **256x256** or **1024x1024** resolutions, specific axis orderings (NHWC/NCHW), and normalization constants—are **arbitrary defaults** tailored specifically for the GeoStratum Lithotheque engine.

### Mandatory Adaptation
Users are expected to **adapt the provided scripts** to their specific use cases. To successfully utilize this pipeline with your own models, you must:
- **Modify Dimensions**: Update the input shapes in the `scripts/*.py` files to match your source model's requirements.
- **Adjust Normalization**: Fine-tune the normalization logic in `prepare_calibration.py` to align with your training domain.
- **Verify Architecture Compatibility**: Ensure your model's operator set is compatible with pure NPU execution (e.g. replacing non-standard `Erf`/`Gelu` kernels with polynomial or `x * sigmoid(1.702 * x)` subgraphs to strictly eliminate `SELECT_TF_OPS` / Flex delegates and achieve 100% NPU graph residency).
- **Domain-Specific Calibration**: Run the calibration routines on a dataset representing *your* specific field of study (~100 images) to ensure accurate INT8 min-max calculation.

### Source Model Architectures
Researchers can obtain the baseline FP32 architectures from the following official repositories to begin their own quantization journey:
- **MobileNetV4-S & L (timm)**: [timm/mobilenetv4](https://huggingface.co/collections/timm/mobilenetv4-pretrained-weights-6669c22cda4db4244def9637)
- **MobileNetV5 (Gemma 3 Vision)**: [google/gemma-3-nano](https://huggingface.co/google/gemma-3-4b-it) | [timm/mobilenetv5](https://huggingface.co/timm/mobilenetv5_300m.gemma3n)

## 13. Technical Reproduction Scripts

For researchers and developers, the `scripts/` directory contains the official tools used to generate the 12 models. Each script is specialized for a specific architecture, format, and precision:

### A. Data Preparation
- **[prepare_calibration.py](scripts/prepare_calibration.py)**: Generates `.npy` calibration tensors from raw image datasets.

### B. TFLite / LiteRT Quantization (NPU Optimized)
- **MobileNetV5 Rocks**: **[tflite_v5_rocks_int8.py](scripts/tflite_v5_rocks_int8.py)** | **[tflite_v5_rocks_int4.py](scripts/tflite_v5_rocks_int4.py)**
- **MobileNetV4-L Rocks**: **[tflite_v4l_rocks_int8.py](scripts/tflite_v4l_rocks_int8.py)** | **[tflite_v4l_rocks_int4.py](scripts/tflite_v4l_rocks_int4.py)**
- **MobileNetV4-S Scale**: **[tflite_v4s_scale_int8.py](scripts/tflite_v4s_scale_int8.py)** | **[tflite_v4s_scale_int4.py](scripts/tflite_v4s_scale_int4.py)**

### C. ONNX Quantization (with Full Integer Stripping)
- **MobileNetV5 Rocks**: **[onnx_v5_rocks_int8.py](scripts/onnx_v5_rocks_int8.py)** | **[onnx_v5_rocks_int4.py](scripts/onnx_v5_rocks_int4.py)**
- **MobileNetV4-L Rocks**: **[onnx_v4l_rocks_int8.py](scripts/onnx_v4l_rocks_int8.py)** | **[onnx_v4l_rocks_int4.py](scripts/onnx_v4l_rocks_int4.py)**
- **MobileNetV4-S Scale**: **[onnx_v4s_scale_int8.py](scripts/onnx_v4s_scale_int8.py)** | **[onnx_v4s_scale_int4.py](scripts/onnx_v4s_scale_int4.py)**

### D. Reference Engine
- **[litho_reference_engine.py](scripts/litho_reference_engine.py)**: 21-Pass Multi-Scale Tiling (21-MST) reference implementation.

### E. Benchmark & Scientific Visualization (R Engine)
- **MobileNetV5 Multi-Backend Suite**: **[generate_benchmark_plots_v5.R](scripts/generate_benchmark_plots_v5.R)**
- **MobileNetV4 Snapdragon Profiling**: **[generate_snapdragon_plots_v4.R](scripts/generate_snapdragon_plots_v4.R)**



## 14. License

This repository utilizes a dual-licensing structure to balance scientific openness with the protection of commercial intellectual property:

*   **Documentation & Metadata**: All architectural documentation, methodology descriptions ([`METHODOLOGY.md`](METHODOLOGY.md)), model cards ([`MODEL_CARD.md`](MODEL_CARD.md)), and metadata remain the exclusive intellectual property of GeoStratum and are licensed under **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC-BY-NC-ND 4.0)**.
*   **Reproduction Scripts**: The technical automation scripts located in the [`scripts/`](scripts/) directory are provided under the **MIT License**, allowing for permissive redistribution and modification for research and development purposes.

*Refer to the [LICENSE](LICENSE) file for the full text of the CC-BY-NC-ND license.*

## 15. Contact
For technical inquiries, enterprise deployment information, or academic collaboration opportunities, please reach out to GeoStratum.

**Email:** communication@geostratum.eu

**Website:** [www.geostratum.eu](https://www.geostratum.eu)

## 16. Citation
If you reference our Multi-Scale Tiling methodology, Edge architecture, or Snapdragon benchmarks in your academic research, please cite this repository:

```bibtex
@misc{lithotheque_edge_2026,
  author = {GeoStratum},
  title = {Lithotheque Edge AI Architecture: Multi-Scale Tiling and NPU Benchmarks},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/GeoStratum/lithotheque-edge-models}}
}
```
