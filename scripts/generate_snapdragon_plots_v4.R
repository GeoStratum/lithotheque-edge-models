# R Script for Generating Qualcomm Snapdragon X (ARM64) Hardware Benchmark Plots
# Data source: results/benchmark_snapdragon_v4_100_runs.csv

csv_file <- if (file.exists("results/benchmark_snapdragon_v4_100_runs.csv")) "results/benchmark_snapdragon_v4_100_runs.csv" else "benchmark_snapdragon_v4_100_runs.csv"
output_dir <- if (dir.exists("results")) "results" else "."

df <- read.csv(csv_file, stringsAsFactors = FALSE)

# Filter valid numeric runs
df$Latency_num <- suppressWarnings(as.numeric(df$Latency_ms))
df_valid <- df[!is.na(df$Latency_num), ]

# Color palette definition
color_npu <- "#2A9D8F"   # Teal / Energy-efficient NPU
color_cpu <- "#E76F51"   # Coral / Oryon CPU
color_gpu <- "#457B9D"   # Blue / Adreno GPU
grid_color <- "#E0E0E0"

# ---------------------------------------------------------------------------
# 1. COMPARATIVE LATENCY ACROSS BACKENDS AND QUANTIZATIONS (BARPLOT)
# ---------------------------------------------------------------------------
png(filename = file.path(output_dir, "5_mobilenetv4_snapdragon_latency_comparison.png"), width = 1100, height = 650, res = 130)
par(mar = c(5, 5, 4, 2), bg = "#FAFAFA")

means <- c(
  mean(df_valid$Latency_num[df_valid$Format == "W4A8" & df_valid$Backend == "NPU"]),
  mean(df_valid$Latency_num[df_valid$Format == "INT8" & df_valid$Backend == "NPU"]),
  mean(df_valid$Latency_num[df_valid$Format == "INT8" & df_valid$Backend == "CPU"]),
  mean(df_valid$Latency_num[df_valid$Format == "W4A8" & df_valid$Backend == "CPU"])
)

labels <- c("NPU (W4A8)\nHexagon HTP", "NPU (INT8)\nHexagon HTP", "CPU (INT8)\nOryon ARM64", "CPU (W4A8)\nOryon ARM64")
bar_colors <- c(color_npu, "#1D3557", color_cpu, "#F4A261")

bp <- barplot(means, names.arg = labels, col = bar_colors, border = NA,
              ylim = c(0, max(means) * 1.25),
              main = "MobileNetV4: Inference Latency by Snapdragon X Accelerator",
              ylab = "Average Latency (ms / image - lower is better)",
              font.main = 2, col.lab = "#333333", col.axis = "#555555")
grid(col = grid_color, lty = "dotted")
barplot(means, names.arg = NA, col = bar_colors, border = NA, ylim = c(0, max(means) * 1.25), yaxt = "n", xaxt = "n", add = TRUE)


fps <- round(1000 / means, 1)
text(bp, means + 0.7, labels = paste0(round(means, 2), " ms\n(", fps, " fps)"), font = 2, cex = 0.9, col = "#222222")
abline(h = 16.67, col = "#888888", lty = "dashed", lwd = 1.2) # 60 FPS real-time threshold
text(x = 1.0, y = 17.5, labels = "60 FPS Threshold (16.67 ms)", col = "#666666", cex = 0.8, font = 3)

legend("topleft", 
       legend = c("Hexagon NPU (HTP)", "Oryon CPU", "60 FPS Real-time Threshold"),
       fill = c(color_npu, color_cpu, NA),
       border = c(NA, NA, NA),
       lty = c(NA, NA, 2), col = c(NA, NA, "#888888"), lwd = c(NA, NA, 1.2),
       bg = "white", box.col = "#DDDDDD")
dev.off()

# ---------------------------------------------------------------------------
# 2. INFERENCE STABILITY ACROSS 100 RUNS (NPU, CPU & DISPERSION)
# ---------------------------------------------------------------------------
png(filename = file.path(output_dir, "6_mobilenetv4_snapdragon_stability_100_runs.png"), width = 1600, height = 600, res = 130)
par(mfrow = c(1, 3), mar = c(5, 5, 4, 2), bg = "#FAFAFA")

# A. Sequential latency trajectory across 100 NPU runs
runs_w4a8_npu <- df_valid$Latency_num[df_valid$Format == "W4A8" & df_valid$Backend == "NPU"]
runs_int8_npu <- df_valid$Latency_num[df_valid$Format == "INT8" & df_valid$Backend == "NPU"]

plot(1:100, runs_int8_npu, type = "l", col = "#1D3557", lwd = 2,
     main = "A. Temporal Stability (NPU)",
     xlab = "Inference ID (Sequential Runs)", ylab = "Latency (ms)",
     ylim = c(0, max(c(runs_int8_npu, runs_w4a8_npu)) * 1.25),
     font.main = 2, col.lab = "#333333", col.axis = "#555555")
grid(col = grid_color, lty = "dotted")
lines(1:100, runs_w4a8_npu, col = color_npu, lwd = 2.5)
points(1:100, runs_w4a8_npu, pch = 19, col = color_npu, cex = 0.6)
legend("topright", legend = c("NPU W4A8 (2.29 ms)", "NPU INT8 (2.67 ms)"),
       col = c(color_npu, "#1D3557"), lwd = 2.5, pch = c(19, NA),
       bg = "white", box.col = "#DDDDDD", cex = 0.85)

# B. Sequential latency trajectory across 100 CPU runs (Oryon ARM64)
runs_w4a8_cpu <- df_valid$Latency_num[df_valid$Format == "W4A8" & df_valid$Backend == "CPU"]
runs_int8_cpu <- df_valid$Latency_num[df_valid$Format == "INT8" & df_valid$Backend == "CPU"]

plot(1:100, runs_w4a8_cpu, type = "l", col = "#F4A261", lwd = 2,
     main = "B. Temporal Stability (CPU Oryon)",
     xlab = "Inference ID (Sequential Runs)", ylab = "Latency (ms)",
     ylim = c(0, max(c(runs_int8_cpu, runs_w4a8_cpu)) * 1.25),
     font.main = 2, col.lab = "#333333", col.axis = "#555555")
grid(col = grid_color, lty = "dotted")
lines(1:100, runs_int8_cpu, col = color_cpu, lwd = 2.5)
points(1:100, runs_int8_cpu, pch = 19, col = color_cpu, cex = 0.6)
legend("topright", legend = c("CPU INT8 (9.46 ms)", "CPU W4A8 (18.12 ms)"),
       col = c(color_cpu, "#F4A261"), lwd = 2.5, pch = c(19, NA),
       bg = "white", box.col = "#DDDDDD", cex = 0.85)

# C. Latency distribution boxplot (NPU vs CPU)
data_boxplot <- list(
  "NPU\nW4A8" = runs_w4a8_npu,
  "NPU\nINT8" = runs_int8_npu,
  "CPU\nINT8" = runs_int8_cpu,
  "CPU\nW4A8" = runs_w4a8_cpu
)

boxplot(data_boxplot, col = c(color_npu, "#1D3557", color_cpu, "#F4A261"),
        border = "#333333", outline = TRUE,
        main = "C. Dispersion & Variance (100 Runs)",
        ylab = "Latency (ms - log scale)", log = "y",
        font.main = 2, col.lab = "#333333", col.axis = "#555555")
grid(col = grid_color, lty = "dotted")
dev.off()

# ---------------------------------------------------------------------------
# 3. EXECUTIVE SNAPDRAGON X BENCHMARK DASHBOARD
# ---------------------------------------------------------------------------
png(filename = file.path(output_dir, "7_mobilenetv4_snapdragon_executive_dashboard.png"), width = 1350, height = 750, res = 130)
layout(matrix(c(1, 2, 3, 3), nrow = 2, byrow = TRUE), heights = c(1.2, 0.8))
par(bg = "#FAFAFA")

# Panel 1: Processing throughput (FPS)
par(mar = c(4, 5, 3.5, 2))
fps_vals <- c(fps[1], fps[2], fps[3], fps[4])
bp_fps <- barplot(fps_vals, names.arg = c("NPU W4A8", "NPU INT8", "CPU INT8", "CPU W4A8"),
                  col = c(color_npu, "#1D3557", color_cpu, "#F4A261"), border = NA,
                  ylim = c(0, max(fps_vals) * 1.2),
                  main = "1. Processing Throughput (Frames / Second)",
                  ylab = "Frames per Second (FPS)", font.main = 2)
grid(col = grid_color, lty = "dotted")
barplot(fps_vals, names.arg = NA, col = c(color_npu, "#1D3557", color_cpu, "#F4A261"), border = NA, ylim = c(0, max(fps_vals) * 1.2), yaxt = "n", xaxt = "n", add = TRUE)
text(bp_fps, fps_vals + 20, labels = paste0(round(fps_vals, 0), " FPS"), font = 2, cex = 0.95, col = "#222222")

# Panel 2: Acceleration speedups relative to CPU
par(mar = c(4, 5, 3.5, 2))
speedup_w4a8 <- mean(df_valid$Latency_num[df_valid$Format == "W4A8" & df_valid$Backend == "CPU"]) / mean(runs_w4a8_npu)
speedup_int8 <- mean(df_valid$Latency_num[df_valid$Format == "INT8" & df_valid$Backend == "CPU"]) / mean(runs_int8_npu)
speedups <- c(speedup_w4a8, speedup_int8)

bp_sp <- barplot(speedups, names.arg = c("W4A8 (NPU vs CPU)", "INT8 (NPU vs CPU)"),
                 col = c(color_npu, "#1D3557"), border = NA,
                 ylim = c(0, max(speedups) * 1.3),
                 main = "2. NPU Hardware Acceleration Factor",
                 ylab = "Speedup Factor (x)", font.main = 2)
grid(col = grid_color, lty = "dotted")
barplot(speedups, names.arg = NA, col = c(color_npu, "#1D3557"), border = NA, ylim = c(0, max(speedups) * 1.3), yaxt = "n", xaxt = "n", add = TRUE)

text(bp_sp, speedups + 0.5, labels = paste0(round(speedups, 1), "x faster"), font = 2, cex = 0.95, col = "#222222")

# Panel 3: Executive summary text & hardware specifications
par(mar = c(1, 2, 2, 2))
plot.new()
rect(0, 0, 1, 1, col = "white", border = "#DDDDDD", lwd = 1.5)
text(0.5, 0.85, "QUALCOMM SNAPDRAGON X HARDWARE QUALIFICATION SUMMARY", font = 2, cex = 1.1, col = "#1D3557")
text(0.5, 0.65, "Processor: Qualcomm Oryon CPU (10 Cores ARM64 @ 3.40 GHz) | NPU: Qualcomm Hexagon HTP v2.37.1 (45 TOPS)", cex = 0.9, col = "#444444")
text(0.5, 0.45, "• MobileNetV4 W4A8 on NPU: 2.29 ms (~436 fps) - Ultra-low latency for continuous real-time vision without throttling.", cex = 0.9, col = "#2A9D8F", font = 2)
text(0.5, 0.25, "• MobileNetV4 INT8 on NPU: 2.67 ms (~374 fps) - Optimal trade-off for geological accuracy (78.78% Top-1 / 91.37% Top-3).", cex = 0.9, col = "#1D3557", font = 2)
text(0.5, 0.08, "• Comparison with MobileNetV5-300M: MobileNetV4 is 168x to 208x faster on NPU (ideal for mobile field survey).", cex = 0.85, col = "#666666", font = 3)

dev.off()

cat("Snapdragon benchmark plots successfully generated in:", output_dir, "\n")
