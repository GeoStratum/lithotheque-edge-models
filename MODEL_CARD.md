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
* **Accuracy:** 96.76% (Top-1) on the Legacy backend, 79.87% on Standard/Premium quantized backends.
* **Latency:** Ranging from 18.4 ms (Dedicated NPU) to 142.8 ms (Legacy CPU).
* *For comprehensive hardware profiling, please refer to the main `README.md`.*

## Limitations & Bias
* **Occlusion & Surface Conditions:** Accuracy degrades if the rock sample is heavily covered in mud, moss, or artificial coatings. Freshly broken surfaces yield the most accurate predictions.
* **Lighting Dependency:** Extreme low-light conditions or harsh direct camera flash may alter texture perception, slightly increasing the margin of error.
* **Micro-crystalline vs. Cryptocrystalline:** Some visually identical rocks require chemical or microscopic analysis for definitive differentiation. The model predicts the highest mathematical probability based strictly on visual morphological features.
