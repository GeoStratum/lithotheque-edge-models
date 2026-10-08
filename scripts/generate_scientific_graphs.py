import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

RESULTS_DIR = r"E:\fine tuning\rock identifier\lithotheque-edge-models\results"
CSV_PATH = os.path.join(RESULTS_DIR, "rtx5060_scientific_analysis.csv")

df = pd.read_csv(CSV_PATH)
print("Données chargées:")
print(df.groupby(["Format", "Backend"]).agg({"Latency_ms": ["count", "mean", "std", "min", "max"]}))

# ==============================================================================
# Graphique 1: Latence Comparée par Backend et Format (Diagramme en Barres d'Erreur)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

summary = df.groupby(["Format", "Backend"])["Latency_ms"].agg(["mean", "std"]).reset_index()
# Ordonner logiquement
order = [
    ("INT8", "Tensor_Cores_INT8"),
    ("INT8", "CPU_x64"),
    ("W4A8", "Tensor_Cores_INT4"),
    ("W4A8", "CPU_x64")
]
summary["order"] = summary.apply(lambda r: order.index((r["Format"], r["Backend"])) if (r["Format"], r["Backend"]) in order else 99, axis=1)
summary = summary.sort_values("order")

labels = [
    "INT8\n(RTX 5060 Tensor Cores)",
    "INT8\n(Intel Core i3 CPU)",
    "W4A8\n(RTX 5060 Tensor Cores)",
    "W4A8\n(Intel Core i3 CPU)"
]
means = summary["mean"].values
stds = summary["std"].values
colors = ["#10b981", "#ef4444", "#3b82f6", "#f59e0b"]

bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, edgecolor="black", alpha=0.88, width=0.55)

for bar, mean in zip(bars, means):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 80, f"{mean:.1f} ms", ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel("Inference Latency (ms) - Lower is Better", fontsize=12, fontweight='bold')
ax.set_title("MobileNetV5 (512x512) - Inference Latency Across Backends\nNVIDIA GeForce RTX 5060 (Blackwell) vs Intel Core i3-12100F", fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, max(means) * 1.22)
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
chart1_path = os.path.join(RESULTS_DIR, "latency_comparison_rtx5060.png")
fig.savefig(chart1_path)
plt.close(fig)
print(f"Graphique 1 généré: {chart1_path}")

# ==============================================================================
# Graphique 2: Distribution et Stabilité des 100 Runs (Boxplot / Violin)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

data_groups = []
labels_box = []
for fmt, bk, lbl in [
    ("INT8", "Tensor_Cores_INT8", "INT8 Tensor Cores\n(RTX 5060)"),
    ("W4A8", "Tensor_Cores_INT4", "W4A8 Tensor Cores\n(RTX 5060)"),
    ("INT8", "CPU_x64", "INT8 CPU\n(i3-12100F)"),
    ("W4A8", "CPU_x64", "W4A8 CPU\n(i3-12100F)")
]:
    sub = df[(df["Format"] == fmt) & (df["Backend"] == bk)]["Latency_ms"].values
    data_groups.append(sub)
    labels_box.append(lbl)

try:
    bplot = ax.boxplot(data_groups, patch_artist=True, tick_labels=labels_box, medianprops=dict(color='black', linewidth=2))
except TypeError:
    bplot = ax.boxplot(data_groups, patch_artist=True, labels=labels_box, medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax.set_ylabel("Inference Latency (ms)", fontsize=12, fontweight='bold')
ax.set_title("Empirical Latency Stability Across Iterations (Variance & Outliers)\nMobileNetV5 Rock Classification Pipeline", fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
chart2_path = os.path.join(RESULTS_DIR, "latency_stability_distribution.png")
fig.savefig(chart2_path)
plt.close(fig)
print(f"Graphique 2 généré: {chart2_path}")

# ==============================================================================
# Graphique 3: Compromis Pareto (Latence vs Accuracy Top-1)
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

pts = [
    {"label": "INT8 (RTX 5060 Tensor Cores)", "lat": df[(df["Format"] == "INT8") & (df["Backend"] == "Tensor_Cores_INT8")]["Latency_ms"].mean(), "acc": 83.33, "color": "#10b981", "size": 180},
    {"label": "W4A8 (RTX 5060 Tensor Cores)", "lat": df[(df["Format"] == "W4A8") & (df["Backend"] == "Tensor_Cores_INT4")]["Latency_ms"].mean(), "acc": 80.33, "color": "#3b82f6", "size": 180},
    {"label": "INT8 (Intel CPU x64)", "lat": df[(df["Format"] == "INT8") & (df["Backend"] == "CPU_x64")]["Latency_ms"].mean(), "acc": 83.33, "color": "#ef4444", "size": 140},
    {"label": "W4A8 (Intel CPU x64)", "lat": df[(df["Format"] == "W4A8") & (df["Backend"] == "CPU_x64")]["Latency_ms"].mean(), "acc": 80.33, "color": "#f59e0b", "size": 140},
    {"label": "Snapdragon CPU INT8 (Oryon ARM64)", "lat": 5695.84, "acc": 83.33, "color": "#8b5cf6", "size": 140},
    {"label": "Snapdragon CPU W4A8 (Oryon ARM64)", "lat": 7119.80, "acc": 80.33, "color": "#ec4899", "size": 140}
]

for p in pts:
    ax.scatter(p["lat"], p["acc"], color=p["color"], s=p["size"], edgecolors="black", zorder=4)
    # Décaler le texte pour la lisibilité
    offset_y = 0.25 if "W4A8" in p["label"] else -0.35
    offset_x = 40 if p["lat"] < 2000 else -250
    ax.annotate(p["label"], (p["lat"], p["acc"]), textcoords="offset points", xytext=(10, 8 if "INT8" in p["label"] else -16),
                fontsize=9.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=p["color"], alpha=0.9))

ax.set_xlabel("Inference Latency (ms) - Log/Linear Scale", fontsize=12, fontweight='bold')
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight='bold')
ax.set_title("Pareto Efficiency: Geological Accuracy vs Inference Latency\nComparison Across RTX 5060 & Snapdragon Environments", fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(78.5, 85.5)
ax.set_xlim(-200, 7800)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
chart3_path = os.path.join(RESULTS_DIR, "accuracy_vs_latency_pareto.png")
fig.savefig(chart3_path)
plt.close(fig)
print(f"Graphique 3 généré: {chart3_path}")
print("Tous les graphiques scientifiques ont été générés avec succès !")
