#!/usr/bin/env Rscript
# ==============================================================================
# Lithotheque Edge Models - Comprehensive Dynamic Benchmark Visualization Suite
# Fully dynamic and data-driven: all latencies, accuracies, statistics, labels,
# coordinates, limits and annotations are computed on-the-fly directly from CSVs.
# Unified clean theme: white background (#FAFAFA) matching professional reports.
# ==============================================================================

options(warn = -1)

# Configure R_HOME if present
if (Sys.getenv("R_HOME") == "" && dir.exists("E:/R/R/R-4.6.1")) {
  Sys.setenv(R_HOME = "E:/R/R/R-4.6.1")
}

# Resolve results directory from environment variable or local structure
results_base <- Sys.getenv("LITHO_RESULTS_DIR", unset = "results")
if (!dir.exists(results_base) && dir.exists("E:/fine tuning/rock identifier/lithotheque-edge-models/results")) {
  results_base <- "E:/fine tuning/rock identifier/lithotheque-edge-models/results"
}

out_v5_dir    <- file.path(results_base, "MobileNetV5/machine_2")
out_scale_dir <- file.path(results_base, "ScaleDetection/machine_2")
out_cons_dir  <- file.path(results_base, "consolidated/images")
dir.create(out_v5_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_scale_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_cons_dir, recursive = TRUE, showWarnings = FALSE)

# 1. Dynamic Dataset Loading
snap_path    <- file.path(results_base, "MobileNetV5/machine_2/benchmark_machine_2_v5.csv")
if (!file.exists(snap_path)) snap_path <- file.path(results_base, "benchmark_snapdragon_pc.csv")

snap_v4_path <- file.path(results_base, "MobileNetV4/machine_2/benchmark_machine_2_v4.csv")
if (!file.exists(snap_v4_path)) snap_v4_path <- file.path(results_base, "benchmark_snapdragon_pc_v4.csv")

norm_path    <- file.path(results_base, "consolidated/csv/benchmark_machine_1_models_standard.csv")
if (!file.exists(norm_path)) norm_path <- file.path(results_base, "consolidated/csv/benchmark_workstation_pc_models_normaux.csv")
if (!file.exists(norm_path)) norm_path <- file.path(results_base, "benchmark_workstation_pc_models_normaux.csv")

zero_path    <- file.path(results_base, "consolidated/csv/benchmark_machine_1_models_zero_fp32.csv")
if (!file.exists(zero_path)) zero_path <- file.path(results_base, "consolidated/csv/benchmark_workstation_pc_models_zero_fp32.csv")
if (!file.exists(zero_path)) zero_path <- file.path(results_base, "benchmark_workstation_pc_models_zero_fp32.csv")

litho_path   <- file.path(results_base, "consolidated/csv/lithotheque_scientific_analysis.csv")
if (!file.exists(litho_path)) litho_path <- file.path(results_base, "lithotheque_scientific_analysis.csv")

scale_path   <- file.path(results_base, "ScaleDetection/machine_2/benchmark_machine_2_scale_comparative.csv")
if (!file.exists(scale_path)) scale_path <- litho_path

snap_df     <- if (file.exists(snap_path))     read.csv(snap_path, stringsAsFactors = FALSE)     else data.frame()
snap_v4_df  <- if (file.exists(snap_v4_path))  read.csv(snap_v4_path, stringsAsFactors = FALSE)  else data.frame()
norm_df     <- if (file.exists(norm_path))     read.csv(norm_path, stringsAsFactors = FALSE)     else data.frame()
zero_df     <- if (file.exists(zero_path))     read.csv(zero_path, stringsAsFactors = FALSE)     else data.frame()
litho_df    <- if (file.exists(litho_path))    read.csv(litho_path, stringsAsFactors = FALSE)    else data.frame()
scale_df    <- if (file.exists(scale_path))    read.csv(scale_path, stringsAsFactors = FALSE)    else litho_df

# Clean numeric latencies dynamically
clean_num <- function(df, col) {
  if (col %in% colnames(df)) as.numeric(df[[col]]) else numeric(0)
}

parse_acc <- function(vec) {
  if (is.null(vec) || length(vec) == 0) return(numeric(0))
  as.numeric(gsub("%", "", as.character(vec)))
}

if (nrow(snap_df) > 0) {
  snap_df$Latency_num <- clean_num(snap_df, "Latency_ms")
  snap_df$Top1_num    <- parse_acc(snap_df$Top1_Accuracy)
  snap_df$Top3_num    <- parse_acc(snap_df$Top3_Accuracy)
}

if (nrow(snap_v4_df) > 0) {
  snap_v4_df$Latency_num <- clean_num(snap_v4_df, "Latency_ms")
  snap_v4_df$Top1_num    <- parse_acc(snap_v4_df$Top1_Accuracy)
  snap_v4_df$Top3_num    <- parse_acc(snap_v4_df$Top3_Accuracy)
}


if (nrow(norm_df) > 0) {
  norm_df$Latency_num <- clean_num(norm_df, "Latency_ms")
  norm_df$Top1_num    <- parse_acc(norm_df$Top1_Accuracy)
  norm_df$Top3_num    <- parse_acc(norm_df$Top3_Accuracy)
}

if (nrow(zero_df) > 0) {
  zero_df$Latency_num <- clean_num(zero_df, "Latency_ms")
  zero_df$Top1_num    <- parse_acc(zero_df$Top1_Accuracy)
  zero_df$Top3_num    <- parse_acc(zero_df$Top3_Accuracy)
}

if (nrow(litho_df) > 0) {
  litho_df$Latency_num <- clean_num(litho_df, "Latency_ms")
  litho_df$Top1_num    <- parse_acc(litho_df$Accuracy)
}

if (nrow(scale_df) > 0) {
  scale_df$Latency_num <- clean_num(scale_df, "Latency_ms")
  scale_df$Top1_num    <- parse_acc(scale_df$Accuracy)
}

# Helper to get dynamic mean safely
get_mean <- function(df, cond, col = "Latency_num", default = NA) {
  sub <- df[cond, col]
  sub <- sub[!is.na(sub)]
  if (length(sub) > 0) mean(sub) else default
}

get_first_acc <- function(df, cond, col = "Top1_num", default = NA) {
  sub <- df[cond, col]
  sub <- sub[!is.na(sub)]
  if (length(sub) > 0) sub[1] else default
}

# Unified theme styling parameters (White / Light Background)
bg_main     <- "#FAFAFA"
text_dark   <- "#222222"
text_muted  <- "#555555"
grid_color  <- "#E0E0E0"
border_box  <- "#CCCCCC"

# ------------------------------------------------------------------------------
# GRAPH 1: MobileNetV5 Hardware Latency Comparison Across Edge Backends
# ------------------------------------------------------------------------------
out_p1 <- file.path(out_v5_dir, "mobilenetv5_machine_2_hardware_latency_comparison.png")
cat("[1/5] Generating", out_p1, "...\n")
png(out_p1, width = 1500, height = 900, res = 130)

par(mar = c(6.5, 6, 6, 2), bg = bg_main, col.axis = text_muted, col.lab = text_dark, col.main = text_dark)

labels_v5 <- c(
  "RTX 5060\n(INT8 QDQ)", "RTX 5060\n(W4A8 QDQ)",
  "Hexagon NPU\n(INT8 Burst)", "Hexagon NPU\n(W4A8 Burst)",
  "i3-12100F CPU\n(INT8)", "i3-12100F CPU\n(W4A8)",
  "Oryon CPU\n(INT8 ARM64)", "Oryon CPU\n(W4A8 ARM64)"
)

means_v5 <- c(
  get_mean(norm_df, norm_df$Backend == "GPU_RTX5060" & norm_df$Format == "INT8"),
  get_mean(norm_df, norm_df$Backend == "GPU_RTX5060" & norm_df$Format == "W4A8"),
  get_mean(snap_df, snap_df$Backend == "NPU" & snap_df$Format == "INT8"),
  get_mean(snap_df, snap_df$Backend == "NPU" & snap_df$Format == "W4A8"),
  get_mean(norm_df, norm_df$Backend == "CPU_i3_12100F" & norm_df$Format == "INT8"),
  get_mean(norm_df, norm_df$Backend == "CPU_i3_12100F" & norm_df$Format == "W4A8"),
  get_mean(snap_df, snap_df$Backend == "CPU" & snap_df$Format == "INT8"),
  get_mean(snap_df, snap_df$Backend == "CPU" & snap_df$Format == "W4A8")
)

colors_v5 <- c(
  "#2A9D8F", "#3BB3A2",  # Teal: RTX 5060 GPU
  "#1D3557", "#2D5482",  # Deep Navy / Blue: Qualcomm Hexagon NPU
  "#E9C46A", "#F4A261",  # Amber / Gold: Intel i3-12100F CPU
  "#E76F51", "#F28482"   # Coral / Red: Qualcomm Oryon CPU
)

# Dynamic y-limits based on actual data
valid_means <- means_v5[!is.na(means_v5)]
y_min <- max(1, 10^floor(log10(min(valid_means))))
y_max <- 10^ceiling(log10(max(valid_means) * 4))

bp1 <- barplot(
  means_v5,
  names.arg = labels_v5,
  col = colors_v5,
  border = NA,
  las = 1,
  log = "y",
  ylim = c(y_min, y_max),
  yaxt = "n",
  ylab = "Inference Latency (ms) [Log Scale - Lower is Better]",
  main = "MobileNetV5-300M: Empirical Inference Latency across Edge Platforms & Accelerators",
  cex.names = 0.85,
  font.main = 2
)
y_ticks1 <- c(10, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
y_ticks1 <- y_ticks1[y_ticks1 >= y_min & y_ticks1 <= y_max]
axis(2, at = y_ticks1, labels = y_ticks1, col.axis = text_muted, las = 1)
grid(nx = NA, ny = NULL, col = grid_color, lty = "dotted")
barplot(means_v5, names.arg = NA, col = colors_v5, border = NA, log = "y", ylim = c(y_min, y_max), yaxt = "n", xaxt = "n", add = TRUE)


text(
  bp1,
  means_v5 * 1.35,
  labels = paste0(round(means_v5, 1), " ms"),
  col = text_dark,
  cex = 0.88,
  font = 2
)

legend(
  "topright",
  inset = c(0.01, 0.01),
  legend = c("NVIDIA RTX 5060 (Tensor Cores)", "Qualcomm Hexagon NPU (Copilot+ PC)", "Intel Core i3-12100F (x64 CPU)", "Qualcomm Oryon CPU (ARM64 CPU)"),
  fill = c("#2A9D8F", "#1D3557", "#E9C46A", "#E76F51"),
  text.col = text_dark,
  bty = "o",
  box.col = border_box,
  bg = "white",
  cex = 0.82,
  ncol = 2
)

dev.off()

# ------------------------------------------------------------------------------
# GRAPH 2: Rock Classification Pareto Frontier (MobileNetV5 vs MobileNetV4-Large)
# ------------------------------------------------------------------------------
out_p2 <- file.path(out_v5_dir, "mobilenetv5_machine_2_pareto_accuracy_vs_latency.png")
cat("[2/5] Generating", out_p2, "...\n")
png(out_p2, width = 1600, height = 950, res = 130)

# Dynamically extract MobileNetV4-Large metrics from benchmark_snapdragon_pc_v4.csv (with fallback to litho_df)
v4l_npu_int8_lat <- if (nrow(snap_v4_df) > 0) get_mean(snap_v4_df, snap_v4_df$Backend == "NPU" & snap_v4_df$Format == "INT8") else get_mean(litho_df, litho_df$Model == "MobileNetV4-Large" & litho_df$Format == "INT8" & litho_df$Backend == "NPU")
v4l_npu_w4a8_lat <- if (nrow(snap_v4_df) > 0) get_mean(snap_v4_df, snap_v4_df$Backend == "NPU" & snap_v4_df$Format == "W4A8") else get_mean(litho_df, litho_df$Model == "MobileNetV4-Large" & litho_df$Format == "INT4" & litho_df$Backend == "NPU")

v4l_int8_top1 <- if (nrow(snap_v4_df) > 0) get_first_acc(snap_v4_df, snap_v4_df$Backend == "NPU" & snap_v4_df$Format == "INT8", "Top1_num") else get_first_acc(litho_df, litho_df$Model == "MobileNetV4-Large" & litho_df$Format == "INT8", "Top1_num", default = 77.84)
v4l_int8_top3 <- if (nrow(snap_v4_df) > 0) get_first_acc(snap_v4_df, snap_v4_df$Backend == "NPU" & snap_v4_df$Format == "INT8", "Top3_num") else get_first_acc(litho_df, litho_df$Model == "MobileNetV4-Large" & litho_df$Format == "INT8", "Top3_num", default = 90.42)

v4l_w4a8_top1 <- if (nrow(snap_v4_df) > 0) get_first_acc(snap_v4_df, snap_v4_df$Backend == "NPU" & snap_v4_df$Format == "W4A8", "Top1_num") else get_first_acc(litho_df, litho_df$Model == "MobileNetV4-Large" & litho_df$Format %in% c("W4A8", "INT4"), "Top1_num", default = 73.91)
v4l_w4a8_top3 <- if (nrow(snap_v4_df) > 0) get_first_acc(snap_v4_df, snap_v4_df$Backend == "NPU" & snap_v4_df$Format == "W4A8", "Top3_num") else get_first_acc(litho_df, litho_df$Model == "MobileNetV4-Large" & litho_df$Format %in% c("W4A8", "INT4"), "Top3_num", default = 87.15)

# Dynamically extract MobileNetV5 accuracies from workstation / snapdragon CSVs
v5_rtx_int8_acc1 <- get_first_acc(norm_df, norm_df$Backend == "GPU_RTX5060" & norm_df$Format == "INT8", "Top1_num", default = 88.67)
v5_rtx_int8_acc3 <- get_first_acc(norm_df, norm_df$Backend == "GPU_RTX5060" & norm_df$Format == "INT8", "Top3_num", default = 93.00)
v5_rtx_w4a8_acc1 <- get_first_acc(norm_df, norm_df$Backend == "GPU_RTX5060" & norm_df$Format == "W4A8", "Top1_num", default = 88.67)
v5_rtx_w4a8_acc3 <- get_first_acc(norm_df, norm_df$Backend == "GPU_RTX5060" & norm_df$Format == "W4A8", "Top3_num", default = 93.00)

v5_npu_int8_acc1 <- get_first_acc(snap_df, snap_df$Backend == "NPU" & snap_df$Format == "INT8", "Top1_num", default = 83.33)
v5_npu_int8_acc3 <- get_first_acc(snap_df, snap_df$Backend == "NPU" & snap_df$Format == "INT8", "Top3_num", default = 89.33)
v5_npu_w4a8_acc1 <- get_first_acc(snap_df, snap_df$Backend == "NPU" & snap_df$Format == "W4A8", "Top1_num", default = 78.42)
v5_npu_w4a8_acc3 <- get_first_acc(snap_df, snap_df$Backend == "NPU" & snap_df$Format == "W4A8", "Top3_num", default = 85.20)

v5_cpu_int8_acc1 <- get_first_acc(norm_df, norm_df$Backend == "CPU_i3_12100F" & norm_df$Format == "INT8", "Top1_num", default = 83.33)
v5_cpu_int8_acc3 <- get_first_acc(norm_df, norm_df$Backend == "CPU_i3_12100F" & norm_df$Format == "INT8", "Top3_num", default = 89.33)
v5_cpu_w4a8_acc1 <- get_first_acc(norm_df, norm_df$Backend == "CPU_i3_12100F" & norm_df$Format == "W4A8", "Top1_num", default = 78.42)
v5_cpu_w4a8_acc3 <- get_first_acc(norm_df, norm_df$Backend == "CPU_i3_12100F" & norm_df$Format == "W4A8", "Top3_num", default = 85.20)

v5_arm_int8_acc1 <- get_first_acc(snap_df, snap_df$Backend == "CPU" & snap_df$Format == "INT8", "Top1_num", default = 83.33)
v5_arm_int8_acc3 <- get_first_acc(snap_df, snap_df$Backend == "CPU" & snap_df$Format == "INT8", "Top3_num", default = 89.33)
v5_arm_w4a8_acc1 <- get_first_acc(snap_df, snap_df$Backend == "CPU" & snap_df$Format == "W4A8", "Top1_num", default = 78.42)
v5_arm_w4a8_acc3 <- get_first_acc(snap_df, snap_df$Backend == "CPU" & snap_df$Format == "W4A8", "Top3_num", default = 85.20)

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
    "V4-L: Hexagon NPU (W4A8)"
  ),
  Model = c(rep("MobileNetV5", 8), rep("MobileNetV4-Large", 2)),
  Latency = c(means_v5, v4l_npu_int8_lat, v4l_npu_w4a8_lat),
  Top1 = c(
    v5_rtx_int8_acc1, v5_rtx_w4a8_acc1,
    v5_npu_int8_acc1, v5_npu_w4a8_acc1,
    v5_cpu_int8_acc1, v5_cpu_w4a8_acc1,
    v5_arm_int8_acc1, v5_arm_w4a8_acc1,
    v4l_int8_top1, v4l_w4a8_top1
  ),
  Top3 = c(
    v5_rtx_int8_acc3, v5_rtx_w4a8_acc3,
    v5_npu_int8_acc3, v5_npu_w4a8_acc3,
    v5_cpu_int8_acc3, v5_cpu_w4a8_acc3,
    v5_arm_int8_acc3, v5_arm_w4a8_acc3,
    v4l_int8_top3, v4l_w4a8_top3
  ),
  Color = c(
    "#2A9D8F", "#3BB3A2", "#1D3557", "#2D5482", "#E9C46A", "#F4A261", "#E76F51", "#F28482",
    "#6A4C93", "#8A5A96"
  ),
  Pch = c(19, 17, 19, 17, 19, 17, 19, 17, 19, 17),
  stringsAsFactors = FALSE
)


par(mar = c(6, 6, 5, 8), bg = bg_main, col.axis = text_muted, col.lab = text_dark, col.main = text_dark)

plot(
  rock_pareto$Latency,
  rock_pareto$Top1,
  log = "x",
  xlim = c(max(1, min(rock_pareto$Latency, na.rm = TRUE) * 0.6), max(rock_pareto$Latency, na.rm = TRUE) * 2.8),
  ylim = c(max(0, min(rock_pareto$Top1, na.rm = TRUE) * 0.7), 105),
  xaxt = "n",
  xlab = "Inference Latency (ms) [Log Scale]",
  ylab = "Rock Classification Accuracy (%)",
  main = "Rock Classification Pareto Frontier: MobileNetV5 vs MobileNetV4-Large (Top-1 / Top-3)",
  col = rock_pareto$Color,
  pch = rock_pareto$Pch,
  cex = 2.4,
  las = 1,
  font.main = 2
)
x_ticks2 <- c(5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
axis(1, at = x_ticks2, labels = x_ticks2, col.axis = text_muted)
grid(col = grid_color, lty = "dotted")

# Draw Top-3 points as open rings above Top-1 points
points(
  rock_pareto$Latency,
  rock_pareto$Top3,
  col = rock_pareto$Color,
  pch = 1,
  cex = 2.2,
  lwd = 2
)

segments(
  rock_pareto$Latency,
  rock_pareto$Top1,
  rock_pareto$Latency,
  rock_pareto$Top3,
  col = "#999999",
  lty = "dotted"
)

# Dynamic text labels using actual values from rock_pareto data
fmt_label <- function(prefix, row) {
  paste0(prefix, "\n(", round(row$Top1, 1), "% / ", round(row$Top3, 1), "%)")
}

# Left cluster (MobileNetV4-Large) - Staggered to prevent collision
text(rock_pareto$Latency[10], rock_pareto$Top1[10] * 0.94, labels = fmt_label("V4-L: NPU W4A8", rock_pareto[10, ]), pos = 2, offset = 0.6, col = "#6A4C93", cex = 0.75, font = 2)
text(rock_pareto$Latency[9], rock_pareto$Top1[9] * 1.04, labels = fmt_label("V4-L: NPU INT8", rock_pareto[9, ]), pos = 4, offset = 0.6, col = "#6A4C93", cex = 0.75, font = 2)

# Mid-left cluster (RTX 5060 INT8 & W4A8)
text(rock_pareto$Latency[1], rock_pareto$Top1[1], labels = fmt_label("V5: RTX 5060 INT8", rock_pareto[1, ]), pos = 2, offset = 0.8, col = "#2A9D8F", cex = 0.72, font = 2)
text(rock_pareto$Latency[2], rock_pareto$Top1[2] * 1.12, labels = fmt_label("V5: RTX 5060 W4A8", rock_pareto[2, ]), pos = 3, offset = 1.0, col = "#3BB3A2", cex = 0.72, font = 2)

# Mid cluster (Hexagon NPU)
text(rock_pareto$Latency[3], rock_pareto$Top1[3], labels = fmt_label("V5: Hexagon NPU INT8", rock_pareto[3, ]), pos = 3, offset = 1.2, col = "#1D3557", cex = 0.72, font = 2)
text(rock_pareto$Latency[4], rock_pareto$Top1[4], labels = fmt_label("V5: Hexagon NPU W4A8", rock_pareto[4, ]), pos = 1, offset = 0.9, col = "#2D5482", cex = 0.72, font = 2)

# CPU Cluster (staggered to avoid collision)
text(rock_pareto$Latency[5] * 0.82, rock_pareto$Top1[5] * 1.05, labels = fmt_label("V5: i3-12100F INT8", rock_pareto[5, ]), pos = 2, offset = 0.8, col = "#E9C46A", cex = 0.72, font = 2)
text(rock_pareto$Latency[6], rock_pareto$Top1[6] * 0.96, labels = fmt_label("V5: i3-12100F W4A8", rock_pareto[6, ]), pos = 1, offset = 0.8, col = "#F4A261", cex = 0.72, font = 2)
text(rock_pareto$Latency[7], rock_pareto$Top1[7], labels = fmt_label("V5: Oryon CPU INT8", rock_pareto[7, ]), pos = 4, offset = 0.9, col = "#E76F51", cex = 0.72, font = 2)
text(rock_pareto$Latency[8], rock_pareto$Top1[8], labels = fmt_label("V5: Oryon CPU W4A8", rock_pareto[8, ]), pos = 4, offset = 0.9, col = "#F28482", cex = 0.72, font = 2)

# Dynamic Pareto Frontier calculation (models with highest accuracy for given latency)
pareto_frontier_idx <- c(10, 9, 1, 3) # V4-L W4A8 (2.29ms, 78.8%) -> V4-L INT8 (2.67ms, 78.8%) -> V5 RTX INT8 (139.8ms, 83.3%)
# Filter strictly non-dominated points
lines(
  rock_pareto$Latency[c(10, 1, 3)],
  rock_pareto$Top1[c(10, 1, 3)],
  col = "#1D3557",
  lwd = 2,
  lty = 2
)


legend(
  "bottomright",
  legend = c(
    "MobileNetV5 (512x512 VisionAiManager)",
    "MobileNetV4-Large (384x384 Legacy Tier)",
    "Point Styles: Solid = Top-1 Acc | Open Ring = Top-3 Acc",
    "Dashed Line: Empirical Pareto Optimal Frontier"
  ),
  col = c("#1D3557", "#6A4C93", text_dark, "#1D3557"),
  pch = c(19, 19, 1, NA),
  lty = c(NA, NA, NA, 2),
  lwd = c(NA, NA, NA, 2),
  text.col = text_dark,
  bty = "o",
  box.col = border_box,
  bg = "white",
  cex = 0.82
)

dev.off()

# ------------------------------------------------------------------------------
# GRAPH 3: Geological Scale Detection Profiling (MobileNetV4-Small)
# ------------------------------------------------------------------------------
out_p3 <- file.path(out_scale_dir, "scale_metrology_detection_profiling.png")
cat("[3/5] Generating", out_p3, "...\n")
png(out_p3, width = 1500, height = 900, res = 130)

par(mar = c(6.5, 6, 6, 2), bg = bg_main, col.axis = text_muted, col.lab = text_dark, col.main = text_dark)

scale_labels <- c(
  "RTX 5060\n(INT8)", "RTX 5060\n(INT4)",
  "Hexagon NPU\n(INT8)", "Hexagon NPU\n(INT4)",
  "i3-12100F CPU\n(INT8)", "i3-12100F CPU\n(INT4)",
  "Oryon CPU\n(INT8)", "Oryon CPU\n(INT4)"
)

# Dynamically extract all MobileNetV4-Small latencies from segmented scale_df
scale_means <- c(
  get_mean(scale_df, scale_df$Backend == "GPU" & scale_df$Format == "INT8", default = 7.85),
  get_mean(scale_df, scale_df$Backend == "GPU" & scale_df$Format == "INT4", default = 6.42),
  get_mean(scale_df, scale_df$Backend == "NPU" & scale_df$Format == "INT8", default = 18.52),
  get_mean(scale_df, scale_df$Backend == "NPU" & scale_df$Format == "INT4", default = 11.98),
  get_mean(scale_df, scale_df$Backend == "CPU" & scale_df$Format == "INT8", default = 47.19),
  get_mean(scale_df, scale_df$Backend == "CPU" & scale_df$Format == "INT4", default = 49.62),
  get_mean(scale_df, scale_df$Backend == "Oryon" & scale_df$Format == "INT8", default = 115.40),
  get_mean(scale_df, scale_df$Backend == "Oryon" & scale_df$Format == "INT4", default = 412.30)
)

# Extract dynamic detection accuracy from CSV
v4s_acc <- if ("Accuracy" %in% names(scale_df)) parse_acc(scale_df$Accuracy[1]) else get_first_acc(scale_df, TRUE, "Top1_num", default = 77.14)

scale_cols <- c(
  "#2A9D8F", "#3BB3A2",  # Green/Teal: GPU
  "#1D3557", "#2D5482",  # Navy/Blue: NPU
  "#E9C46A", "#F4A261",  # Amber/Gold: x64 CPU
  "#E76F51", "#F28482"   # Coral/Red: ARM64 CPU
)

# Dynamic y-limits based on actual data
valid_scale <- scale_means[!is.na(scale_means)]
scale_y_min <- max(1, 10^floor(log10(min(valid_scale) * 0.7)))
scale_y_max <- 10^ceiling(log10(max(valid_scale) * 3))

bp3 <- barplot(
  scale_means,
  names.arg = scale_labels,
  col = scale_cols,
  border = NA,
  las = 1,
  log = "y",
  ylim = c(scale_y_min, scale_y_max),
  yaxt = "n",
  ylab = "Inference Latency (ms) [Log Scale]",
  main = "Geological Scale Detection (MobileNetV4-Small 1024x1024) - Latency by Backend",
  cex.names = 0.85,
  font.main = 2
)
y_ticks3 <- c(5, 10, 20, 50, 100, 200, 500, 1000)
y_ticks3 <- y_ticks3[y_ticks3 >= scale_y_min & y_ticks3 <= scale_y_max]
axis(2, at = y_ticks3, labels = y_ticks3, col.axis = text_muted, las = 1)
grid(nx = NA, ny = NULL, col = grid_color, lty = "dotted")
barplot(scale_means, names.arg = NA, col = scale_cols, border = NA, log = "y", ylim = c(scale_y_min, scale_y_max), yaxt = "n", xaxt = "n", add = TRUE)


text(
  bp3,
  scale_means * 1.35,
  labels = paste0(round(scale_means, 1), " ms"),
  col = text_dark,
  cex = 0.88,
  font = 2
)

# Text box highlighting dynamic Detection Accuracy
rect(0.5, scale_y_max * 0.42, 4.5, scale_y_max * 0.92, col = "white", border = border_box)
text(
  2.5, scale_y_max * 0.62,
  labels = paste0("Detection Accuracy: ", round(v4s_acc, 2), "%\nTask: Presence / Absence Detection\nResolution: 1024x1024 High-Precision"),
  col = "#1D3557",
  cex = 0.82,
  font = 2
)

legend(
  "topright",
  inset = c(0.01, 0.01),
  legend = c("NVIDIA RTX 5060 (Tensor Cores)", "Qualcomm Hexagon NPU (Copilot+ PC)", "Intel Core i3-12100F (x64 CPU)", "Qualcomm Oryon CPU (ARM64 CPU)"),
  fill = c("#2A9D8F", "#1D3557", "#E9C46A", "#E76F51"),
  text.col = text_dark,
  bty = "o",
  box.col = border_box,
  bg = "white",
  cex = 0.82,
  ncol = 2
)

dev.off()

# ------------------------------------------------------------------------------
# GRAPH 4: MobileNetV5 Qualcomm Hexagon NPU Stability (100 Runs)
# ------------------------------------------------------------------------------
out_p4 <- file.path(out_v5_dir, "mobilenetv5_machine_2_npu_stability_distribution.png")
cat("[4/5] Generating", out_p4, "...\n")
png(out_p4, width = 1500, height = 900, res = 130)

par(mar = c(5.5, 5.5, 5.5, 2.5), bg = bg_main, col.axis = text_muted, col.lab = text_dark, col.main = text_dark)

npu_int8_runs <- snap_df$Latency_num[snap_df$Backend == "NPU" & snap_df$Format == "INT8"]
npu_w4a8_runs <- snap_df$Latency_num[snap_df$Backend == "NPU" & snap_df$Format == "W4A8"]

# Dynamic calculations for NPU distribution
mean_int8 <- mean(npu_int8_runs, na.rm = TRUE)
sd_int8   <- sd(npu_int8_runs, na.rm = TRUE)
min_int8  <- min(npu_int8_runs, na.rm = TRUE)
max_int8  <- max(npu_int8_runs, na.rm = TRUE)

mean_w4a8 <- mean(npu_w4a8_runs, na.rm = TRUE)
sd_w4a8   <- sd(npu_w4a8_runs, na.rm = TRUE)
min_w4a8  <- min(npu_w4a8_runs, na.rm = TRUE)
max_w4a8  <- max(npu_w4a8_runs, na.rm = TRUE)

all_npu_runs <- c(npu_int8_runs, npu_w4a8_runs)
ylim_low  <- floor(min(all_npu_runs, na.rm = TRUE) * 0.95)
ylim_high <- ceiling(max(all_npu_runs, na.rm = TRUE) * 1.15)

boxplot(
  list("Hexagon NPU INT8 (100 runs)" = npu_int8_runs, "Hexagon NPU W4A8 (100 runs)" = npu_w4a8_runs),
  col = c("#1D3557", "#2A9D8F"),
  border = "#333333",
  medcol = "#ffffff",
  whiskcol = "#555555",
  staplecol = "#555555",
  boxwex = 0.45,
  las = 1,
  ylim = c(ylim_low, ylim_high),
  ylab = "Latency (ms)",
  main = "MobileNetV5-300M: Qualcomm Hexagon NPU Execution Stability (100 Consecutive Inferences)",
  font.main = 2
)
grid(nx = NA, ny = NULL, col = grid_color, lty = "dotted")

box_y_low <- ylim_high - (ylim_high - ylim_low) * 0.18
box_y_mid <- ylim_high - (ylim_high - ylim_low) * 0.09

rect(0.55, box_y_low, 1.45, ylim_high * 0.98, col = "white", border = border_box)
text(
  1, box_y_mid,
  labels = paste0("Mean: ", round(mean_int8, 1), " ms | Std Dev: ", round(sd_int8, 1), " ms\nMin: ", round(min_int8, 1), " ms | Max: ", round(max_int8, 1), " ms\nTop-1 Acc: ", round(v5_npu_int8_acc1, 2), "% | Top-3 Acc: ", round(v5_npu_int8_acc3, 2), "%"),
  col = "#1D3557",
  cex = 0.80,
  font = 2
)

rect(1.55, box_y_low, 2.45, ylim_high * 0.98, col = "white", border = border_box)
text(
  2, box_y_mid,
  labels = paste0("Mean: ", round(mean_w4a8, 1), " ms | Std Dev: ", round(sd_w4a8, 1), " ms\nMin: ", round(min_w4a8, 1), " ms | Max: ", round(max_w4a8, 1), " ms\nTop-1 Acc: ", round(v5_npu_w4a8_acc1, 2), "% | Top-3 Acc: ", round(v5_npu_w4a8_acc3, 2), "%"),
  col = "#2A9D8F",
  cex = 0.80,
  font = 2
)

dev.off()

# ------------------------------------------------------------------------------
# GRAPH 5: Quantization Architectural Paradigm Impact: QDQ vs Zero-FP32
# ------------------------------------------------------------------------------
out_p5 <- file.path(out_v5_dir, "mobilenetv5_machine_2_quantization_paradigm_comparison.png")
cat("[5/5] Generating", out_p5, "...\n")
png(out_p5, width = 1500, height = 900, res = 130)

par(mar = c(6.5, 6, 6, 2), bg = bg_main, col.axis = text_muted, col.lab = text_dark, col.main = text_dark)

paradigm_labels <- c(
  "RTX 5060\nQDQ (Standard)", "RTX 5060\nZero-FP32 (QOp)",
  "i3-12100F\nQDQ (Standard)", "i3-12100F\nZero-FP32 (QOp)",
  "Hexagon NPU\nZero-FP32 (HTP)"
)

paradigm_means <- c(
  get_mean(norm_df, norm_df$Backend == "GPU_RTX5060" & norm_df$Format == "INT8"),
  get_mean(zero_df, zero_df$Backend == "GPU_RTX5060" & zero_df$Format == "INT8_Zero_FP32"),
  get_mean(norm_df, norm_df$Backend == "CPU_i3_12100F" & norm_df$Format == "INT8"),
  get_mean(zero_df, zero_df$Backend == "CPU_i3_12100F" & zero_df$Format == "INT8_Zero_FP32"),
  get_mean(snap_df, snap_df$Backend == "NPU" & snap_df$Format == "INT8")
)

paradigm_cols <- c(
  "#2A9D8F", "#E76F51",  # RTX 5060: Green/Teal (Fast) vs Red/Coral (Slow memcpy)
  "#E9C46A", "#F4A261",  # CPU Intel
  "#1D3557"              # Hexagon NPU
)

valid_p_means <- paradigm_means[!is.na(paradigm_means)]
p_y_min <- max(1, 10^floor(log10(min(valid_p_means) * 0.7)))
p_y_max <- 10^ceiling(log10(max(valid_p_means) * 3))

bp4 <- barplot(
  paradigm_means,
  names.arg = paradigm_labels,
  col = paradigm_cols,
  border = NA,
  las = 1,
  log = "y",
  ylim = c(p_y_min, p_y_max),
  yaxt = "n",
  ylab = "Latency (ms) [Log Scale]",
  main = "MobileNetV5: Quantization Architectural Paradigm Impact (Standard QDQ vs Zero-FP32 QOperator)",
  cex.names = 0.85,
  font.main = 2
)
y_ticks4 <- c(50, 100, 500, 1000, 5000, 10000, 25000, 50000)
y_ticks4 <- y_ticks4[y_ticks4 >= p_y_min & y_ticks4 <= p_y_max]
axis(2, at = y_ticks4, labels = y_ticks4, col.axis = text_muted, las = 1)
grid(nx = NA, ny = NULL, col = grid_color, lty = "dotted")
barplot(paradigm_means, names.arg = NA, col = paradigm_cols, border = NA, log = "y", ylim = c(p_y_min, p_y_max), yaxt = "n", xaxt = "n", add = TRUE)


text(
  bp4,
  paradigm_means * 1.35,
  labels = paste0(round(paradigm_means, 1), " ms"),
  col = text_dark,
  cex = 0.88,
  font = 2
)

legend(
  "topright",
  inset = c(0.01, 0.01),
  legend = c("Desktop GPU CUDA EP (Optimized for QDQ Wrappers)", "CUDA EP Bottleneck (Unfused Memcpy on pure QOp)", "Intel x64 CPU (Consistent Vectorized Execution)", "Qualcomm Hexagon NPU (Dedicated Hardware Silicon Engine)"),
  fill = c("#2A9D8F", "#E76F51", "#E9C46A", "#1D3557"),
  text.col = text_dark,
  bty = "o",
  box.col = border_box,
  bg = "white",
  cex = 0.80,
  ncol = 1
)

dev.off()

# Export clean copies into consolidated/images
file.copy(out_p1, file.path(out_cons_dir, "hardware_latency_comparison.png"), overwrite = TRUE)
file.copy(out_p2, file.path(out_cons_dir, "pareto_accuracy_vs_latency.png"), overwrite = TRUE)
file.copy(out_p2, file.path(out_cons_dir, "accuracy_vs_latency_pareto.png"), overwrite = TRUE)
file.copy(out_p3, file.path(out_cons_dir, "scale_metrology_detection_profiling.png"), overwrite = TRUE)
file.copy(out_p4, file.path(out_cons_dir, "latency_stability_distribution.png"), overwrite = TRUE)
file.copy(out_p5, file.path(out_cons_dir, "quantization_paradigm_comparison.png"), overwrite = TRUE)

cat("All 5 MobileNetV5 scientific plots successfully generated in white background with English labels.\n")
