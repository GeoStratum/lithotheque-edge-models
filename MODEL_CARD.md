# Model Card: Lithotheque Edge AI Models

## Model Details
* **Developer:** GeoStratum
* **Model Versions:** `VisionAiManager` (MobileNetV5-300m / MobileNetV4-Large) and `LocalAiManager` (Metrology).
* **Model Type:** Convolutional Neural Networks (CNN) optimized for Edge Inference (INT8/INT4 Quantization).
* **License:** Proprietary (GeoStratum). 
* **Training Data:** Hybrid corpus of proprietary metrology images and open-source MIT-licensed geological datasets (Udayl, Stealth Technologies).

## Intended Use
* **Primary Use Case:** Offline, on-device geological and lithological classification for field geologists, students, and enthusiasts.
* **Secondary Use Case:** Scale detection and topographical metrology mapping via reference objects.
* **Target Environment:** Mobile devices operating in remote areas without internet connectivity.

## Out-of-Scope Use
These models are designed for educational and preliminary field classification. They **MUST NOT** be used for:
* **Critical Engineering Decisions:** Assessing rock mechanics, stability, or load-bearing capacities for civil engineering or construction.
* **Financial/Mining Exploration:** Making high-stakes financial decisions regarding resource extraction or ore grade estimation.
* **Safety:** Detecting hazardous environments or toxic minerals for safety compliance. 
*Always consult laboratory analysis (e.g., XRD, XRF) for definitive geological validation.*

## Metrics & Performance
* **Accuracy (Standardized Multi-Source Benchmark):**
  * **MobileNetV5-300M (Premium / Standard Tier):** Up to **88.33% Top-1 / 92.67% Top-3** in baseline FP16/BF16, **83.33% Top-1 / 89.33% Top-3** in **INT8 (W8A8)**, and **80.33% Top-1 / 87.67% Top-3** in **W4A8 Full Integer**.
  * **MobileNetV4-Large (Balanced / Legacy Tier):** **78.78% Top-1 / 91.37% Top-3** in baseline FP16/BF16, **77.84% Top-1 / 90.42% Top-3** in **INT8**, and **73.91% Top-1 / 87.15% Top-3** in **W4A8** (reaching >91% Top-1 on in-situ daylight field subsets).
* **Latency:** Sub-3 ms on dedicated NPU hardware for MobileNetV4-Large (~400 FPS steady-state), and 12–18 ms for 1024×1024 Scale Detection.
* *For comprehensive hardware profiling, please refer to the main `README.md`.*

## Limitations & Bias
* **Occlusion & Surface Conditions:** Accuracy degrades if the rock sample is heavily covered in mud, moss, or artificial coatings. Freshly broken surfaces yield the most accurate predictions.
* **Lighting Dependency:** Extreme low-light conditions or harsh direct camera flash may alter texture perception, slightly increasing the margin of error.
* **Micro-crystalline vs. Cryptocrystalline:** Some visually identical rocks require chemical or microscopic analysis for definitive differentiation. The model predicts the highest mathematical probability based strictly on visual morphological features.
