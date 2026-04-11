# Lithotheque Edge Models - GeoStratum
**Technical showcase of the Edge AI architecture powering the Lithotheque offline application.**

![Offline](https://img.shields.io/badge/Inference-100%25_Offline-success)
![Languages](https://img.shields.io/badge/Supported_Languages-14-blue)
![Engine](https://img.shields.io/badge/Engine-LiteRT_Standalone-orange)

## 1. Overview
This repository documents the advanced Edge AI architecture integrated into [GeoStratum Lithotheque](https://www.geostratum.eu/lithotheque). To provide instant, on-device geological classification and scale detection without requiring an internet connection, the application relies on a highly optimized, dual-model ecosystem deployed via **Google Play Asset Delivery (On-Demand)**.

## 2. Dual-Model Ecosystem
The application utilizes two distinct neural networks, fine-tuned specifically for geological field operations:

### A. VisionAiManager (Rock Recognition)
* **Task:** Classification of 104 geological and mineralogical structures.
* **Base Architectures:** `MobileNetV5-300m` for modern devices, and `MobileNetV4-Large` for legacy fallback.
* **Training Corpus:** Fine-tuned on a proprietary dataset of >14,500 validated images.
* **Innovation: Multi-Scale Tiling (21-Pass Strategy):** To capture both macroscopic textures and microscopic crystals without losing data during downscaling, the engine evaluates the image through 21 parallel passes:
  * **1 Global Pass:** Overall sample context.
  * **4 Medium Passes (2x2 Grid):** Regional texture analysis.
  * **16 Fine Passes (4x4 Grid):** Micro-detail and mineral extraction.
  *(Scores are aggregated via weighted average for the final prediction).*
### Dataset Categorization (104 Classes)

The model is trained on a diverse set of 104 geological structures, categorized below by their primary genetic classification.

#### 1. Sedimentary (Sed)
* Rocks formed by the accumulation and lithification of sediments or chemical precipitation.
> Bauxite, Caliche, Chalk, Chert, Clay, Coal, Conglomerate, Coquine, Diatomite, Dolomitic Limestone, Dolomites, Flint, Fossiliferous Limestone, Gypsum, Halite, Limestone, Novaculite, Oolitic Limestone, Phosphate, Potash, Sandstone, Shale (Mudstone), Shale-(Mudstone), Siliceous-sinter, Siltstone, Sodium carbonate, Tufa.

#### 2. Magmatic / Plutonic (Mag)
* Intrusive igneous rocks formed from slowly cooling magma deep underground.
> Anorthosite, Aplite, Diorite, Dolerite, Dunite, Essexite, Gabbro, Granite, Granodiorite, Norite, Pegmatite, Syenite.

#### 3. Volcanic / Extrusive (Volc)
* Extrusive igneous rocks formed from rapidly cooling lava at or near the surface.
> Andesite, Basalt, Dacite, Ignimbrite, Komatiite, Obsidian, Olivine basalt, Olivine-basalt, Phonolite, Pillow (Lava), Pumice, Rhyolite, Tephrite, Trachyte, Tuff, Volcanic bombs.

---
*Note for precision: The dataset also includes the following categories to support specific field identification tasks:*

#### 4. Metamorphic (Meta)
* Rocks altered by extreme heat, pressure, or hydrothermal processes.
> Anthracite, Breccia (Tectonic/Fault), Gneiss, Hornfels, Lapis lazuli, Marble, Phyllite, Quartzite, Schists, Serpentine, Skarn, Slate.

#### 5. Native Elements, Minerals & Ores
* Pure elements, economic ores, and individual rock-forming minerals.
> Bornite, Calcite, Chromite, Cobalt, Columbite-tantalite, Copper, Feldspar, Fluorite, Gold, Iron ore, Labradorite, Lead, Lithium, Magnetite, Malachite, Mariposite, Mica, Molybdenum, Nickel, Platinum, Pyrite, Quartz, Silica, Silver, Sodalite, Stibnite, Sulfur, Tantalum, Tungsten, Uranium, Vanadium, Zeolite, Zinc.

### B. LocalAiManager (Scale Detection)
* **Task:** Metrology assistant detecting reference objects (coins, geological scales).
* **Base Architecture:** `MobileNetV4-Small`.
* **Resolution:** High-res 1024x1024 processing for precise small-object anchoring.
* **Training Corpus:** Fine-tuned on >1,000 reference images.

## 3. Inference Engine & Hardware Fallback
To prevent thermal throttling during the intensive 21-pass analysis, all image preprocessing (resizing, normalization) is written in **Native C++ (JNI) utilizing NEON SIMD instructions**, yielding a 5x to 10x speedup over standard Android pipelines with minimal memory overhead.

Inference is powered by **LiteRT Standalone** with a robust 6-level hardware fallback system to ensure maximum compatibility across the fragmented Android ecosystem:
1. **Dedicated NPU** (e.g., Snapdragon 8 Gen 2+, Google Tensor)
2. **NNAPI** (Android Hardware Acceleration)
3. **Modern GPU** (Shader acceleration)
4. **Modern CPU** (SIMD/Neon vectorization)
5. **Legacy GPU**
6. **XNNPACK** (Highly optimized CPU fallback)

## 4. Deployment Strategy (AI Tiers)
Models are aggressively quantized and delivered dynamically based on the device's hardware profile upon first launch:

| Tier | Hardware Criteria | Quantization | Base Architecture |
| :--- | :--- | :---: | :--- |
| **Premium** | RAM > 6GB + Modern NPU | **INT4** | MobileNetV5 (300m) & V4 (Small) |
| **Standard** | RAM > 6GB | **INT8** | MobileNetV5 (300m) & V4 (Small) |
| **Legacy** | Older / Entry-level devices | **INT8** | MobileNetV4 (Large) & V4 (Small) |

## 5. Visual Evaluation & Benchmarks & Hardware Profiling

All performance metrics and initialization times are highly dependent on the host hardware. To ensure rigorous benchmarking, measurements are conducted on state-of-the-art ARM64 silicon featuring dedicated AI acceleration.

### 💻 Primary Test Environment: Qualcomm Snapdragon X Plus
Benchmarks were executed natively on the **Snapdragon X Plus (X1P-64-100)** platform running Windows on ARM. This SoC closely mirrors the architectural behavior of flagship Android devices, making it an ideal environment for Edge AI validation.

* **Compute (CPU):** Qualcomm Oryon™ CPU (10 Cores, up to 3.4 GHz)
* **AI Engine (NPU):** Qualcomm Hexagon™ NPU (**45 TOPS**) — *Targeted by the Premium INT4 tier.*
* **Memory:** LPDDR5x (8448 MT/s) — *Critical for evaluating cold-start load times.*
* **OS:** Windows 11 ARM64

---

### A. Memory Footprint & Initialization (Cold Start)
*The following benchmarks measure the initialization time required to load the `.tflite` models into RAM/NPU.*

[... ici, tu remets ton tableau avec les temps de 383ms, 708ms, etc. ...]

*(Insert your performance matrices, confusion matrix, or accuracy charts here)*

The application utilizes **Google Play Asset Delivery** to download the appropriate AI Pack based on the device's capabilities. Below is the exact memory footprint for the standalone `.tflite` models across the three deployment tiers.

*Note: Latency and accuracy benchmarks are actively being compiled for the latest INT4/INT8 build.*

## 5. Benchmarks & Hardware Profiling
*(Insert your performance matrices, confusion matrix, or accuracy charts here)*

The application utilizes **Google Play Asset Delivery** to download the appropriate AI Pack based on the device's capabilities. 

### A. Memory Footprint & Initialization (Cold Start)
Cold start performance is critical for field applications. The following benchmarks measure the initialization time required to load the models into memory.
* **Test Environment:** Snapdragon X Plus X1-P64-100 (Windows).

| Target Tier | Hardware Profile | Format | Rock Model Size | Scale Model Size | Total Footprint | Rock Load Time | Scale Load Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Premium** | Modern NPU | INT4 | 182.9 MB | 1.67 MB | **~184.6 MB** | 383.38 ms | 12.20 ms |
| **Standard** | Normal NPU | INT8 | 287.6 MB | 2.67 MB | **~290.3 MB** | 708.17 ms | 44.95 ms |
| **Legacy** | GPU / CPU | INT8 | 119.9 MB | 2.67 MB | **~122.6 MB** | 179.48 ms | 44.18 ms |

> **Architecture Notes:** > * **Premium Tier:** Achieves excellent load times (383ms) despite its high capacity, thanks to the highly optimized INT4 quantization mapping efficiently to the NPU.
> * **Legacy Tier:** The rock model footprint is intentionally bottlenecked to 119.9 MB. This results in the fastest initialization time (179ms) and prevents Out-Of-Memory (OOM) crashes on older, RAM-constrained devices.

### B. Inference Latency & Accuracy
*Note: Real-time per-frame inference latency (speed of the 21-pass tiling strategy) and Top-1 accuracy benchmarks are actively being compiled for the latest builds.*

| Target Tier | Engine Fallback | Avg. Rock Inference (ms) | Avg. Scale Inference (ms) | Top-1 Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Premium** | Dedicated NPU | *TBD* | *TBD* | *TBD* |
| **Standard** | NNAPI / GPU | *TBD* | *TBD* | *TBD* |
| **Legacy** | CPU (XNNPACK) | *TBD* | *TBD* | *TBD* |

> **Architecture Note:** The **Legacy** pack significantly reduces the `rock_model` footprint (119.9 MB) to prevent Out-Of-Memory (OOM) crashes on older devices, ensuring a stable fallback mechanism without compromising the core metrology functionality (`scale_model`).

## 6. Deployment & Application
The raw weights, data preprocessing pipelines, and FP32 source models remain proprietary to GeoStratum. The inference engine is exclusively accessible through the consumer application.

👉 **[Download and test the app on GeoStratum](https://www.geostratum.eu/lithotheque)**

## 7. Citation
If you reference our Multi-Scale Tiling methodology or Edge architecture in your academic research, please cite this repository:

```bibtex
@misc{lithotheque_edge_2026,
  author = {GeoStratum},
  title = {Lithotheque Edge AI Architecture: Multi-Scale Tiling and NPU Benchmarks},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/](https://github.com/)[Your-Username]/lithotheque-edge-models}}
}
