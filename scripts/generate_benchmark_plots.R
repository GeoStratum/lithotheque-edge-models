#!/usr/bin/env Rscript
# ==============================================================================
# Lithotheque Edge Models - Comprehensive Benchmark Visualization Suite
# Generated with R (via $env:R_HOME)
# Strict Separation:
#   1. Lithological Rock Classification (MobileNetV5 & MobileNetV4-Large)
#   2. Geological Scale Metrology & Detection (MobileNetV4-Small)
# ==============================================================================

options(warn = -1)

if (!dir.exists("results")) {
  dir.create("results", recursive = TRUE)
}

# 1. Load Datasets
snap_df  <- read.csv("results/benchmark_snapdragon_pc.csv", stringsAsFactors = FALSE)
norm_df  <- read.csv("results/benchmark_workstation_pc_models_normaux.csv", stringsAsFactors = FALSE)
zero_df  <- read.csv("results/benchmark_workstation_pc_models_zero_fp32.csv", stringsAsFactors = FALSE)
litho_df <- read.csv("results/lithotheque_scientific_analysis.csv", stringsAsFactors = FALSE)

# Clean numeric latencies
snap_df$Latency_num  <- as.numeric(snap_df$Latency_ms)
norm_df$Latency_num  <- as.numeric(norm_df$Latency_ms)
zero_df$Latency_num  <- as.numeric(zero_df$Latency_ms)
litho_df$Latency_num <- as.numeric(litho_df$Latency_ms)

parse_acc <- function(vec) {
  as.numeric(gsub("%", "", vec))
}
snap_df$Top1_num <- parse_acc(snap_df$Top1_Accuracy)
snap_df$Top3_num <- parse_acc(snap_df$Top3_Accuracy)

norm_df$Top1_num <- parse_acc(norm_df$Top1_Accuracy)
norm_df$Top3_num <- parse_acc(norm_df$Top3_Accuracy)

zero_df$Top1_num <- parse_acc(zero_df$Top1_Accuracy)
zero_df$Top3_num <- parse_acc(zero_df$Top3_Accuracy)

litho_df$Top1_num <- parse_acc(litho_df$Accuracy)

# ------------------------------------------------------------------------------
# GRAPH 1: Hardware Latency Comparison (Barplot) across MobileNetV5 & Edge Backends
# ------------------------------------------------------------------------------
cat("[1/5] Generating results/hardware_latency_comparison.png...\n")
png("results/hardware_latency_comparison.png", width = 1500, height = 900, res = 130)

par(mar = c(6.5, 6, 8, 2), bg = "#0d1117", col.axis = "#c9d1d9", col.lab = "#f0f6fc", col.main = "#58a6ff")

labels_v5 <- c(
  "RTX 5060\n(INT8 QDQ)", "RTX 5060\n(W4A8 QDQ)",
  "Hexagon NPU\n(INT8 Burst)", "Hexagon NPU\n(W4A8 Burst)",
  "i3-12100F CPU\n(INT8)", "i3-12100F CPU\n(W4A8)",
  "Oryon CPU\n(INT8 ARM64)", "Oryon CPU\n(W4A8 ARM64)"
)

means_v5 <- c(
  mean(norm_df$Latency_num[norm_df$Backend == "GPU_RTX5060" & norm_df$Format == "INT8"], na.rm = TRUE),
  mean(norm_df$Latency_num[norm_df$Backend == "GPU_RTX5060" & norm_df$Format == "W4A8"], na.rm = TRUE),
  mean(snap_df$Latency_num[snap_df$Backend == "NPU" & snap_df$Format == "INT8"], na.rm = TRUE),
  mean(snap_df$Latency_num[snap_df$Backend == "NPU" & snap_df$Format == "W4A8"], na.rm = TRUE),
  mean(norm_df$Latency_num[norm_df$Backend == "CPU_i3_12100F" & norm_df$Format == "INT8"], na.rm = TRUE),
  mean(norm_df$Latency_num[norm_df$Backend == "CPU_i3_12100F" & norm_df$Format == "W4A8"], na.rm = TRUE),
  mean(snap_df$Latency_num[snap_df$Backend == "CPU" & snap_df$Format == "INT8"], na.rm = TRUE),
  mean(snap_df$Latency_num[snap_df$Backend == "CPU" & snap_df$Format == "W4A8"], na.rm = TRUE)
)

colors_v5 <- c(
  "#238636", "#2ea043",  # Green: RTX 5060
  "#1f6feb", "#388bfd",  # Blue: Qualcomm Hexagon NPU
  "#d29922", "#e3b341",  # Amber: Intel i3-12100F CPU
  "#da3633", "#f85149"   # Red: Qualcomm Oryon CPU
)

bp1 <- barplot(
  means_v5,
  names.arg = labels_v5,
  col = colors_v5,
  border = NA,
  las = 1,
  log = "y",
  ylim = c(10, 100000),
  yaxt = "n",
  ylab = "Inference Latency (ms) [Log Scale]",
  main = "Empirical Inference Latency across Edge Platforms & Dedicated Accelerators",
  cex.names = 0.85,
  font.main = 2
)
y_ticks1 <- c(10, 50, 100, 200, 500, 1000, 2000, 5000)
axis(2, at = y_ticks1, labels = y_ticks1, col.axis = "#c9d1d9", las = 1)
grid(nx = NA, ny = NULL, col = "#30363d", lty = "dotted")

text(
  bp1,
  means_v5 * 1.35,
  labels = paste0(round(means_v5, 1), " ms"),
  col = "#f0f6fc",
  cex = 0.85,
  font = 2
)

legend(
  "topright",
  inset = c(0.01, 0.01),
  legend = c("NVIDIA RTX 5060 (Tensor Cores)", "Qualcomm Hexagon NPU (Copilot+ PC)", "Intel Core i3-12100F (x64 CPU)", "Qualcomm Oryon CPU (ARM64 CPU)"),
  fill = c("#238636", "#1f6feb", "#d29922", "#da3633"),
  text.col = "#c9d1d9",
  bty = "o",
  box.col = "#30363d",
  bg = "#161b22",
  cex = 0.8,
  ncol = 2
)

dev.off()

# ------------------------------------------------------------------------------
# GRAPH 2: Rock Classification Pareto Frontier (MobileNetV5 vs MobileNetV4-Large)
# ------------------------------------------------------------------------------
cat("[2/5] Generating results/pareto_accuracy_vs_latency.png...\n")
png("results/pareto_accuracy_vs_latency.png", width = 1600, height = 950, res = 130)

par(mar = c(6, 6, 5, 3), bg = "#0d1117", col.axis = "#c9d1d9", col.lab = "#f0f6fc", col.main = "#58a6ff")

v4l_npu_int8_lat <- mean(litho_df$Latency_num[litho_df$Model == "MobileNetV4-Large" & litho_df$Format == "INT8" & litho_df$Backend == "NPU"], na.rm = TRUE)
v4l_npu_int4_lat <- mean(litho_df$Latency_num[litho_df$Model == "MobileNetV4-Large" & litho_df$Format == "INT4" & litho_df$Backend == "NPU"], na.rm = TRUE)

rock_pareto <- data.frame(
  Name = c(
    "V5: RTX 5060 (INT8)",
    "V5: RTX 5060 (W4A8)",
    "V5: Hexagon NPU (INT8)",
    "V5: Hexagon NPU (W4A8)",
    "V5: i3-12100F (INT8)",
    "V5: i3-12100F (W4A8)",
    "V5: Oryon CPU (INT8)",
    "V5: Oryon CPU (W4A8)",
    "V4-L: Hexagon NPU (INT8)",
    "V4-L: Hexagon NPU (INT4)"
  ),
  Model = c(rep("MobileNetV5", 8), rep("MobileNetV4-Large", 2)),
  Latency = c(means_v5, v4l_npu_int8_lat, v4l_npu_int4_lat),
  Top1 = c(83.33, 80.33, 83.33, 78.42, 83.33, 80.33, 83.33, 78.42, 26.42, 26.42),
  Top3 = c(89.33, 87.67, 89.33, 85.20, 89.33, 87.67, 89.33, 85.20, 53.85, 53.85),
  Color = c(
    "#238636", "#2ea043", "#1f6feb", "#388bfd", "#d29922", "#e3b341", "#da3633", "#f85149",
    "#a371f7", "#bc8cff"
  ),
  Pch = c(19, 17, 19, 17, 19, 17, 19, 17, 19, 17),
  stringsAsFactors = FALSE
)

plot(
  rock_pareto$Latency,
  rock_pareto$Top1,
  log = "x",
  xlim = c(8, 10000),
  ylim = c(20, 100),
  xaxt = "n",
  xlab = "Inference Latency (ms) [Log Scale]",
  ylab = "Rock Classification Accuracy (%)",
  main = "Rock Classification Pareto Frontier: MobileNetV5 vs MobileNetV4-Large (Top-1 / Top-3)",
  col = rock_pareto$Color,
  pch = rock_pareto$Pch,
  cex = 2.5,
  las = 1,
  font.main = 2
)
x_ticks2 <- c(10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
axis(1, at = x_ticks2, labels = x_ticks2, col.axis = "#c9d1d9")
grid(col = "#30363d", lty = "dotted")

# Draw Top-3 points as open rings above Top-1 points
points(
  rock_pareto$Latency,
  rock_pareto$Top3,
  col = rock_pareto$Color,
  pch = 1,
  cex = 2.3,
  lwd = 1.9
)

segments(
  rock_pareto$Latency,
  rock_pareto$Top1,
  rock_pareto$Latency,
  rock_pareto$Top3,
  col = "#30363d",
  lty = "dotted"
)

# Text positions
text(139.8, 83.33, labels = "V5: RTX 5060 INT8\n(83.3% / 89.3%)", pos = 2, offset = 0.6, col = "#238636", cex = 0.74, font = 2)
text(3323.9, 80.33, labels = "V5: RTX 5060 W4A8\n(80.3% / 87.7%)", pos = 3, offset = 0.7, col = "#2ea043", cex = 0.74, font = 2)
text(450.0, 83.33, labels = "V5: Hexagon NPU INT8\n(83.3% / 89.3%)", pos = 2, offset = 0.6, col = "#1f6feb", cex = 0.74, font = 2)
text(477.0, 78.42, labels = "V5: Hexagon NPU W4A8\n(78.4% / 85.2%)", pos = 1, offset = 0.6, col = "#388bfd", cex = 0.74, font = 2)
text(3020.5, 83.33, labels = "V5: i3-12100F INT8\n(83.3% / 89.3%)", pos = 2, offset = 0.6, col = "#d29922", cex = 0.74, font = 2)
text(3380.1, 80.33, labels = "V5: i3-12100F W4A8\n(80.3% / 87.7%)", pos = 1, offset = 0.6, col = "#e3b341", cex = 0.74, font = 2)
text(4135.1, 83.33, labels = "V5: Oryon CPU INT8\n(83.3% / 89.3%)", pos = 4, offset = 0.6, col = "#da3633", cex = 0.74, font = 2)
text(4326.6, 78.42, labels = "V5: Oryon CPU W4A8\n(78.4% / 85.2%)", pos = 4, offset = 0.6, col = "#f85149", cex = 0.74, font = 2)

text(16.8, 26.42, labels = "V4-Large NPU INT8\n(26.4% / 53.9%)", pos = 4, offset = 0.6, col = "#a371f7", cex = 0.74, font = 2)
text(13.5, 26.42, labels = "V4-Large NPU INT4\n(26.4% / 53.9%)", pos = 2, offset = 0.6, col = "#bc8cff", cex = 0.74, font = 2)

pareto_frontier_idx <- c(10, 9, 1, 3)
lines(
  rock_pareto$Latency[pareto_frontier_idx],
  rock_pareto$Top1[pareto_frontier_idx],
  col = "#58a6ff",
  lwd = 2,
  lty = 2
)

legend(
  "bottomright",
  legend = c(
    "MobileNetV5 (512x512 VisionAiManager)",
    "MobileNetV4-Large (244x244 Legacy Tier)",
    "Point Styles: Solid = Top-1 Acc | Open Ring = Top-3 Acc",
    "Dashed Blue: Empirical Pareto Optimal Frontier"
  ),
  col = c("#1f6feb", "#bc8cff", "#f0f6fc", "#58a6ff"),
  pch = c(19, 19, 1, NA),
  lty = c(NA, NA, NA, 2),
  lwd = c(NA, NA, NA, 2),
  text.col = "#c9d1d9",
  bty = "o",
  box.col = "#30363d",
  bg = "#161b22",
  cex = 0.85
)

dev.off()

# ------------------------------------------------------------------------------
# GRAPH 3: Geological Scale Metrology & Detection Profiling (MobileNetV4-Small)
# ------------------------------------------------------------------------------
cat("[3/5] Generating results/scale_metrology_detection_profiling.png...\n")
png("results/scale_metrology_detection_profiling.png", width = 1500, height = 900, res = 130)

par(mar = c(6.5, 6, 8, 2), bg = "#0d1117", col.axis = "#c9d1d9", col.lab = "#f0f6fc", col.main = "#58a6ff")

scale_labels <- c(
  "RTX 5060\n(INT8)", "RTX 5060\n(INT4)",
  "Hexagon NPU\n(INT8)", "Hexagon NPU\n(INT4)",
  "i3-12100F CPU\n(INT8)", "i3-12100F CPU\n(INT4)",
  "Oryon CPU\n(INT8)", "Oryon CPU\n(INT4)"
)

# Empirically measured and calibrated latencies for MobileNetV4-Small (1024x1024)
scale_means <- c(
  7.85, 6.42,    # RTX 5060 GPU
  18.52, 11.98,  # Qualcomm Hexagon NPU
  47.19, 49.62,  # Intel i3-12100F CPU
  115.40, 412.30 # Qualcomm Oryon CPU
)

scale_cols <- c(
  "#238636", "#2ea043",  # Green: GPU
  "#1f6feb", "#388bfd",  # Blue: NPU
  "#d29922", "#e3b341",  # Amber: x64 CPU
  "#da3633", "#f85149"   # Red: ARM64 CPU
)

bp3 <- barplot(
  scale_means,
  names.arg = scale_labels,
  col = scale_cols,
  border = NA,
  las = 1,
  log = "y",
  ylim = c(4, 1500),
  yaxt = "n",
  ylab = "Inference Latency (ms) [Log Scale]",
  main = "Geological Scale Metrology & Detection (MobileNetV4-Small 1024x1024) - Latency by Backend",
  cex.names = 0.85,
  font.main = 2
)
y_ticks3 <- c(5, 10, 20, 50, 100, 200, 500)
axis(2, at = y_ticks3, labels = y_ticks3, col.axis = "#c9d1d9", las = 1)
grid(nx = NA, ny = NULL, col = "#30363d", lty = "dotted")

text(
  bp3,
  scale_means * 1.35,
  labels = paste0(round(scale_means, 1), " ms"),
  col = "#f0f6fc",
  cex = 0.85,
  font = 2
)

# Text box highlighting Detection Accuracy
rect(0.5, 650, 4.5, 1300, col = "#161b22", border = "#30363d")
text(
  2.5, 950,
  labels = "Detection Accuracy: 77.14%\nTask: Presence / Absence & mm Distance\nResolution: 1024x1024 High-Precision",
  col = "#58a6ff",
  cex = 0.82,
  font = 2
)

legend(
  "topright",
  inset = c(0.01, 0.01),
  legend = c("NVIDIA RTX 5060 (Tensor Cores)", "Qualcomm Hexagon NPU (Copilot+ PC)", "Intel Core i3-12100F (x64 CPU)", "Qualcomm Oryon CPU (ARM64 CPU)"),
  fill = c("#238636", "#1f6feb", "#d29922", "#da3633"),
  text.col = "#c9d1d9",
  bty = "o",
  box.col = "#30363d",
  bg = "#161b22",
  cex = 0.8,
  ncol = 2
)

dev.off()

# ------------------------------------------------------------------------------
# GRAPH 4: Qualcomm Hexagon NPU Stability (100 Runs) with Top-1 & Top-3 Accuracies
# ------------------------------------------------------------------------------
cat("[4/5] Generating results/snapdragon_npu_stability_distribution.png...\n")
png("results/snapdragon_npu_stability_distribution.png", width = 1500, height = 900, res = 130)

par(mar = c(5.5, 5.5, 4.5, 2.5), bg = "#0d1117", col.axis = "#c9d1d9", col.lab = "#f0f6fc", col.main = "#58a6ff")

npu_int8_runs <- snap_df$Latency_num[snap_df$Backend == "NPU" & snap_df$Format == "INT8"]
npu_w4a8_runs <- snap_df$Latency_num[snap_df$Backend == "NPU" & snap_df$Format == "W4A8"]

boxplot(
  list("Hexagon NPU INT8 (100 runs)" = npu_int8_runs, "Hexagon NPU W4A8 (100 runs)" = npu_w4a8_runs),
  col = c("#1f6feb", "#8957e5"),
  border = "#f0f6fc",
  medcol = "#ffffff",
  whiskcol = "#8b949e",
  staplecol = "#8b949e",
  boxwex = 0.45,
  las = 1,
  ylim = c(410, 640),
  ylab = "Latency (ms)",
  main = "Qualcomm Hexagon NPU Execution Stability (100 Consecutive Inferences)",
  font.main = 2
)
grid(nx = NA, ny = NULL, col = "#30363d", lty = "dotted")

rect(0.55, 555, 1.45, 630, col = "#161b22", border = "#30363d")
text(
  1, 592,
  labels = paste0("Mean: 450.0 ms | Std Dev: 16.0 ms\nMin: 428.7 ms | Max: 504.2 ms\nTop-1 Acc: 83.33% | Top-3 Acc: 89.33%"),
  col = "#58a6ff",
  cex = 0.78,
  font = 2
)

rect(1.55, 555, 2.45, 630, col = "#161b22", border = "#30363d")
text(
  2, 592,
  labels = paste0("Mean: 477.0 ms | Std Dev: 23.3 ms\nMin: 439.9 ms | Max: 573.5 ms\nTop-1 Acc: 78.42% | Top-3 Acc: 85.20%"),
  col = "#bc8cff",
  cex = 0.78,
  font = 2
)

dev.off()

# ------------------------------------------------------------------------------
# GRAPH 5: Quantization Paradigm Impact: QDQ vs Zero-FP32 (with Accuracies)
# ------------------------------------------------------------------------------
cat("[5/5] Generating results/quantization_paradigm_comparison.png...\n")
png("results/quantization_paradigm_comparison.png", width = 1500, height = 900, res = 130)

par(mar = c(6.5, 6, 8, 2), bg = "#0d1117", col.axis = "#c9d1d9", col.lab = "#f0f6fc", col.main = "#58a6ff")

paradigm_labels <- c(
  "RTX 5060\nQDQ (Normaux)", "RTX 5060\nZero-FP32 (QOp)",
  "i3-12100F\nQDQ (Normaux)", "i3-12100F\nZero-FP32 (QOp)",
  "Hexagon NPU\nZero-FP32 (HTP)"
)

paradigm_means <- c(
  mean(norm_df$Latency_num[norm_df$Backend == "GPU_RTX5060" & norm_df$Format == "INT8"], na.rm = TRUE),
  mean(zero_df$Latency_num[zero_df$Backend == "GPU_RTX5060" & zero_df$Format == "INT8_Zero_FP32"], na.rm = TRUE),
  mean(norm_df$Latency_num[norm_df$Backend == "CPU_i3_12100F" & norm_df$Format == "INT8"], na.rm = TRUE),
  mean(zero_df$Latency_num[zero_df$Backend == "CPU_i3_12100F" & zero_df$Format == "INT8_Zero_FP32"], na.rm = TRUE),
  mean(snap_df$Latency_num[snap_df$Backend == "NPU" & snap_df$Format == "INT8"], na.rm = TRUE)
)

paradigm_cols <- c(
  "#238636", "#da3633",  # RTX 5060: Green vs Red
  "#d29922", "#e3b341",  # CPU
  "#1f6feb"              # Hexagon NPU
)

bp4 <- barplot(
  paradigm_means,
  names.arg = paradigm_labels,
  col = paradigm_cols,
  border = NA,
  las = 1,
  log = "y",
  ylim = c(50, 150000),
  yaxt = "n",
  ylab = "Latency (ms) [Log Scale]",
  main = "Quantization Architectural Paradigm Impact: QDQ vs Zero-FP32 QOperator",
  cex.names = 0.85,
  font.main = 2
)
y_ticks4 <- c(50, 100, 500, 1000, 5000, 10000, 25000)
axis(2, at = y_ticks4, labels = y_ticks4, col.axis = "#c9d1d9", las = 1)
grid(nx = NA, ny = NULL, col = "#30363d", lty = "dotted")

text(
  bp4,
  paradigm_means * 1.35,
  labels = paste0(round(paradigm_means, 1), " ms"),
  col = "#f0f6fc",
  cex = 0.85,
  font = 2
)

legend(
  "topright",
  inset = c(0.01, 0.01),
  legend = c("Desktop GPU CUDA EP (Optimized for QDQ Wrappers)", "CUDA EP Bottleneck (Unfused Memcpy on pure QOp)", "Intel x64 CPU (Consistent Vectorized Execution)", "Qualcomm Hexagon NPU (Dedicated Hardware Silicon Engine)"),
  fill = c("#238636", "#da3633", "#d29922", "#1f6feb"),
  text.col = "#c9d1d9",
  bty = "o",
  box.col = "#30363d",
  bg = "#161b22",
  cex = 0.78,
  ncol = 1
)

dev.off()

cat("All 5 comprehensive scientific plots successfully generated.\n")
