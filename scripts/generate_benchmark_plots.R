#!/usr/bin/env Rscript
# ==============================================================================
# Lithotheque Edge Models - Benchmark Visualization Suite
# Generated with R (via $env:R_HOME)
# ==============================================================================

options(warn = -1)

# Ensure results directory exists
if (!dir.exists("results")) {
  dir.create("results", recursive = TRUE)
}

# 1. Load Datasets
snap_df <- read.csv("results/benchmark_snapdragon_pc.csv", stringsAsFactors = FALSE)
norm_df <- read.csv("results/benchmark_workstation_pc_models_normaux.csv", stringsAsFactors = FALSE)
zero_df <- read.csv("results/benchmark_workstation_pc_models_zero_fp32.csv", stringsAsFactors = FALSE)

# Clean numeric latencies
snap_df$Latency_num <- as.numeric(snap_df$Latency_ms)
norm_df$Latency_num <- as.numeric(norm_df$Latency_ms)
zero_df$Latency_num <- as.numeric(zero_df$Latency_ms)

# Clean accuracy
parse_acc <- function(vec) {
  as.numeric(gsub("%", "", vec))
}
snap_df$Top1_num <- parse_acc(snap_df$Top1_Accuracy)
norm_df$Top1_num <- parse_acc(norm_df$Top1_Accuracy)
zero_df$Top1_num <- parse_acc(zero_df$Top1_Accuracy)

# ------------------------------------------------------------------------------
# GRAPH 1: Hardware Latency Comparison (Barplot)
# ------------------------------------------------------------------------------
cat("[1/4] Generating results/hardware_latency_comparison.png...\n")
png("results/hardware_latency_comparison.png", width = 1400, height = 850, res = 130)

par(mar = c(6, 6, 4, 2), bg = "#0d1117", col.axis = "#c9d1d9", col.lab = "#f0f6fc", col.main = "#58a6ff")

# Filter valid hardware measurements for MobileNetV5 INT8 and W4A8
# Backends:
# 1. RTX 5060 (QDQ Standard)
# 2. Snapdragon Hexagon NPU
# 3. Intel i3-12100F CPU
# 4. Snapdragon Oryon CPU
labels <- c(
  "RTX 5060\n(INT8 QDQ)", "RTX 5060\n(W4A8 QDQ)",
  "Hexagon NPU\n(INT8 Burst)", "Hexagon NPU\n(W4A8 Burst)",
  "i3-12100F CPU\n(INT8)", "i3-12100F CPU\n(W4A8)",
  "Oryon CPU\n(INT8 ARM64)", "Oryon CPU\n(W4A8 ARM64)"
)

means <- c(
  mean(norm_df$Latency_num[norm_df$Backend == "GPU_RTX5060" & norm_df$Format == "INT8"], na.rm = TRUE),
  mean(norm_df$Latency_num[norm_df$Backend == "GPU_RTX5060" & norm_df$Format == "W4A8"], na.rm = TRUE),
  mean(snap_df$Latency_num[snap_df$Backend == "NPU" & snap_df$Format == "INT8"], na.rm = TRUE),
  mean(snap_df$Latency_num[snap_df$Backend == "NPU" & snap_df$Format == "W4A8"], na.rm = TRUE),
  mean(norm_df$Latency_num[norm_df$Backend == "CPU_i3_12100F" & norm_df$Format == "INT8"], na.rm = TRUE),
  mean(norm_df$Latency_num[norm_df$Backend == "CPU_i3_12100F" & norm_df$Format == "W4A8"], na.rm = TRUE),
  mean(snap_df$Latency_num[snap_df$Backend == "CPU" & snap_df$Format == "INT8"], na.rm = TRUE),
  mean(snap_df$Latency_num[snap_df$Backend == "CPU" & snap_df$Format == "W4A8"], na.rm = TRUE)
)

colors <- c(
  "#238636", "#2ea043",  # Green for RTX 5060
  "#1f6feb", "#388bfd",  # Blue for Qualcomm Hexagon NPU
  "#d29922", "#e3b341",  # Amber for Intel CPU
  "#da3633", "#f85149"   # Red for Qualcomm Oryon CPU
)

bp <- barplot(
  means,
  names.arg = labels,
  col = colors,
  border = NA,
  las = 1,
  log = "y",
  ylim = c(10, 8000),
  ylab = "Inference Latency (ms) [Log Scale]",
  main = "Empirical Inference Latency across Edge Platforms & Dedicated Accelerators",
  cex.names = 0.85,
  cex.axis = 0.9,
  font.main = 2
)
grid(nx = NA, ny = NULL, col = "#30363d", lty = "dotted")

# Add value labels
text(
  bp,
  means * 1.35,
  labels = paste0(round(means, 1), " ms"),
  col = "#f0f6fc",
  cex = 0.85,
  font = 2
)

legend(
  "topleft",
  legend = c("NVIDIA RTX 5060 (Tensor Cores)", "Qualcomm Hexagon NPU (Copilot+ PC)", "Intel Core i3-12100F (x64 CPU)", "Qualcomm Oryon CPU (ARM64 CPU)"),
  fill = c("#238636", "#1f6feb", "#d29922", "#da3633"),
  text.col = "#c9d1d9",
  bty = "n",
  cex = 0.85
)

dev.off()

# ------------------------------------------------------------------------------
# GRAPH 2: Pareto Efficiency (Accuracy vs Latency)
# ------------------------------------------------------------------------------
cat("[2/4] Generating results/pareto_accuracy_vs_latency.png...\n")
png("results/pareto_accuracy_vs_latency.png", width = 1400, height = 850, res = 130)

par(mar = c(5, 5, 4, 2), bg = "#0d1117", col.axis = "#c9d1d9", col.lab = "#f0f6fc", col.main = "#58a6ff")

# Key hardware-model pairs
pareto_data <- data.frame(
  Name = c(
    "RTX 5060 - INT8 (QDQ)",
    "RTX 5060 - W4A8 (QDQ)",
    "Hexagon NPU - INT8",
    "Hexagon NPU - W4A8",
    "i3-12100F - INT8",
    "i3-12100F - W4A8",
    "Oryon CPU - INT8",
    "Oryon CPU - W4A8"
  ),
  Latency = means,
  Top1 = c(83.33, 80.33, 83.33, 78.42, 83.33, 80.33, 83.33, 78.42),
  Color = c("#238636", "#2ea043", "#1f6feb", "#388bfd", "#d29922", "#e3b341", "#da3633", "#f85149"),
  Pch = c(19, 17, 19, 17, 19, 17, 19, 17),
  stringsAsFactors = FALSE
)

plot(
  pareto_data$Latency,
  pareto_data$Top1,
  log = "x",
  xlim = c(20, 8000),
  ylim = c(75, 86),
  xlab = "Mean Latency (ms) [Log Scale]",
  ylab = "Top-1 Accuracy (%)",
  main = "Pareto Frontier: Accuracy vs. Inference Latency (Rock Identification)",
  col = pareto_data$Color,
  pch = pareto_data$Pch,
  cex = 2.4,
  las = 1,
  font.main = 2
)
grid(col = "#30363d", lty = "dotted")

# Add text annotations
text_offsets_y <- c(0.4, -0.4, 0.4, -0.4, 0.4, -0.4, 0.4, -0.4)
text(
  pareto_data$Latency,
  pareto_data$Top1 + text_offsets_y,
  labels = pareto_data$Name,
  col = "#f0f6fc",
  cex = 0.8,
  font = 2
)

# Connect Pareto optimal frontier (RTX 5060 INT8 -> Hexagon NPU INT8)
lines(
  c(means[1], means[3]),
  c(83.33, 83.33),
  col = "#58a6ff",
  lwd = 2,
  lty = 2
)

legend(
  "bottomleft",
  legend = c("INT8 Quantized (Circle)", "W4A8 Sub-byte (Triangle)", "Pareto Optimal Frontier"),
  col = c("#58a6ff", "#58a6ff", "#58a6ff"),
  pch = c(19, 17, NA),
  lty = c(NA, NA, 2),
  lwd = c(NA, NA, 2),
  text.col = "#c9d1d9",
  bty = "n",
  cex = 0.85
)

dev.off()

# ------------------------------------------------------------------------------
# GRAPH 3: Snapdragon Hexagon NPU Latency Stability Distribution (100 runs)
# ------------------------------------------------------------------------------
cat("[3/4] Generating results/snapdragon_npu_stability_distribution.png...\n")
png("results/snapdragon_npu_stability_distribution.png", width = 1400, height = 850, res = 130)

par(mar = c(5, 5, 4, 2), bg = "#0d1117", col.axis = "#c9d1d9", col.lab = "#f0f6fc", col.main = "#58a6ff")

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
  ylim = c(410, 600),
  ylab = "Latency (ms)",
  main = "Qualcomm Hexagon NPU Execution Stability (100 Consecutive Inferences)",
  font.main = 2
)
grid(nx = NA, ny = NULL, col = "#30363d", lty = "dotted")

# Add summary annotations
text(
  1, 570,
  labels = paste0("Mean: ", round(mean(npu_int8_runs), 1), " ms\nStd Dev: ", round(sd(npu_int8_runs), 1), " ms\nMin: ", round(min(npu_int8_runs), 1), " ms\nMax: ", round(max(npu_int8_runs), 1), " ms"),
  col = "#58a6ff",
  cex = 0.85,
  font = 2
)

text(
  2, 570,
  labels = paste0("Mean: ", round(mean(npu_w4a8_runs), 1), " ms\nStd Dev: ", round(sd(npu_w4a8_runs), 1), " ms\nMin: ", round(min(npu_w4a8_runs), 1), " ms\nMax: ", round(max(npu_w4a8_runs), 1), " ms"),
  col = "#bc8cff",
  cex = 0.85,
  font = 2
)

dev.off()

# ------------------------------------------------------------------------------
# GRAPH 4: Quantization Paradigm Impact: Standard QDQ vs Zero-FP32
# ------------------------------------------------------------------------------
cat("[4/4] Generating results/quantization_paradigm_comparison.png...\n")
png("results/quantization_paradigm_comparison.png", width = 1400, height = 850, res = 130)

par(mar = c(6, 6, 4, 2), bg = "#0d1117", col.axis = "#c9d1d9", col.lab = "#f0f6fc", col.main = "#58a6ff")

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
  "#238636", "#da3633",  # RTX 5060: Green (QDQ fast) vs Red (Memcpy penalty on CUDA EP)
  "#d29922", "#e3b341",  # CPU: Similar performance
  "#1f6feb"              # Hexagon NPU: Native pure integer hardware execution
)

bp2 <- barplot(
  paradigm_means,
  names.arg = paradigm_labels,
  col = paradigm_cols,
  border = NA,
  las = 1,
  log = "y",
  ylim = c(50, 40000),
  ylab = "Latency (ms) [Log Scale]",
  main = "Quantization Architectural Paradigm Impact: QDQ vs Zero-FP32 QOperator",
  cex.names = 0.85,
  cex.axis = 0.9,
  font.main = 2
)
grid(nx = NA, ny = NULL, col = "#30363d", lty = "dotted")

text(
  bp2,
  paradigm_means * 1.5,
  labels = paste0(round(paradigm_means, 1), " ms"),
  col = "#f0f6fc",
  cex = 0.85,
  font = 2
)

legend(
  "topleft",
  legend = c("Desktop GPU CUDA EP (Optimized for FP32/QDQ Wrappers)", "CUDA EP Bottleneck (Unfused CPU/GPU Memcpy on pure QOp)", "Hexagon NPU (Dedicated Pure-Integer Silicon Engine)"),
  fill = c("#238636", "#da3633", "#1f6feb"),
  text.col = "#c9d1d9",
  bty = "n",
  cex = 0.85
)

dev.off()

cat("All 4 scientific plots successfully generated in results/ directory.\n")
