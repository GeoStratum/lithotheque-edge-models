# ==============================================================================
# R Script: Generation of Benchmark Plots for Native Qualcomm Hexagon NPU Binaries
# Standardized on 100 Runs per Model Variant (600 Total Inferences)
# Models: MobileNetV4, MobileNetV5, ScaleDetection_MobileNetV4
# Target Platform: Machine 2 - Qualcomm Snapdragon X Plus (Surface Pro 11 ARM64)
# ==============================================================================

results_base <- Sys.getenv("LITHO_RESULTS_DIR", unset = "results")
if (!dir.exists(results_base) && dir.exists("E:/fine tuning/rock identifier/lithotheque-edge-models/results")) {
  results_base <- "E:/fine tuning/rock identifier/lithotheque-edge-models/results"
}

v4_csv <- file.path(results_base, "MobileNetV4/machine_2/benchmark_machine_2_v4_native_100_runs.csv")
v5_csv <- file.path(results_base, "MobileNetV5/machine_2/benchmark_machine_2_v5_native_100_runs.csv")
scale_csv <- file.path(results_base, "ScaleDetection/machine_2/benchmark_machine_2_scale_native_100_runs.csv")
all_csv <- file.path(results_base, "consolidated/csv/benchmark_machine_2_native_binaries_100_runs.csv")

color_npu_w4 <- "#2A9D8F"   # Teal / W4A8 INT4 NPU
color_npu_int8 <- "#1D3557" # Navy / W8A8 INT8 NPU
grid_color <- "#EAEAEA"

cat("[INFO] Running R Native Snapdragon NPU Plot Generator (100 Runs)...\n")
cat("[INFO] Results Directory:", results_base, "\n")

# ------------------------------------------------------------------------------
# 1. Comparative Latency for All Models on Hexagon NPU (100 Runs AOT .bin)
# ------------------------------------------------------------------------------
if (file.exists(all_csv)) {
  df_all <- read.csv(all_csv, stringsAsFactors = FALSE)
  df_all$Latency_ms <- as.numeric(df_all$Latency_ms)
  
  out_png_all <- file.path(results_base, "consolidated/images/machine_2_native_binaries_comparison.png")
  dir.create(dirname(out_png_all), recursive = TRUE, showWarnings = FALSE)
  
  png(filename = out_png_all, width = 1200, height = 700, res = 130)
  par(mar = c(6, 6, 4, 2), bg = "#FAFAFA")
  
  agg <- aggregate(Latency_ms ~ Model + Format, data = df_all, FUN = mean)
  agg$Label <- paste0(agg$Model, "\n(", agg$Format, ")")
  cols <- ifelse(agg$Format == "W4A8", color_npu_w4, color_npu_int8)
  
  bp <- barplot(agg$Latency_ms, names.arg = agg$Label, col = cols, border = NA,
                ylim = c(0, max(agg$Latency_ms) * 1.25),
                main = "Machine 2 (Hexagon NPU): Standalone Native AOT Invocation Turnaround (100 Runs)",
                ylab = "Turnaround Latency (ms / cold invocation - includes CLI & Disk I/O)",
                font.main = 2, col.lab = "#333333", col.axis = "#555555", las = 1)
  grid(col = grid_color, lty = "dotted")
  barplot(agg$Latency_ms, names.arg = NA, col = cols, border = NA,
          ylim = c(0, max(agg$Latency_ms) * 1.25), yaxt = "n", xaxt = "n", add = TRUE)
  
  fps <- round(1000 / agg$Latency_ms, 1)
  text(bp, agg$Latency_ms + (max(agg$Latency_ms) * 0.03),
       labels = paste0(round(agg$Latency_ms, 1), " ms\n(", fps, " FPS)"),
       font = 2, cex = 0.85, col = "#222222")
  
  legend("topleft",
         legend = c("Hexagon HTP W4A8 AOT (.bin)", "Hexagon HTP W8A8 / INT8 AOT (.bin)"),
         fill = c(color_npu_w4, color_npu_int8), border = NA, bg = "white", box.col = "#DDDDDD")
  dev.off()
  cat("[OK] Generated:", out_png_all, "\n")
}

# ------------------------------------------------------------------------------
# 2. MobileNetV4 Native Stability Plot (100 Consecutive Passes on Hexagon NPU)
# ------------------------------------------------------------------------------
if (file.exists(v4_csv)) {
  df_v4 <- read.csv(v4_csv, stringsAsFactors = FALSE)
  df_v4$Latency_ms <- as.numeric(df_v4$Latency_ms)
  
  out_v4_png <- file.path(results_base, "MobileNetV4/machine_2/mobilenetv4_machine_2_native_stability.png")
  png(filename = out_v4_png, width = 1200, height = 600, res = 130)
  par(mar = c(5, 5, 4, 2), bg = "#FAFAFA")
  
  w4_runs <- df_v4$Latency_ms[df_v4$Format == "W4A8"]
  int8_runs <- df_v4$Latency_ms[df_v4$Format == "INT8"]
  
  plot(1:length(w4_runs), w4_runs, type = "l", col = color_npu_w4, lwd = 2,
       ylim = c(min(c(w4_runs, int8_runs)) * 0.9, max(c(w4_runs, int8_runs)) * 1.15),
       xlab = "Inference Run ID (100 Consecutive Field Images)", ylab = "Hardware Latency (ms)",
       main = "MobileNetV4: Native HTP Stability on Machine 2 (Hexagon NPU, 100 Runs)",
       las = 1, font.main = 2)
  grid(col = grid_color, lty = "dotted")
  lines(1:length(int8_runs), int8_runs, col = color_npu_int8, lwd = 2)
  
  legend("topright",
         legend = c(paste0("W4A8 (Mean: ", round(mean(w4_runs), 1), " ms, p95: ", round(quantile(w4_runs, 0.95), 1), " ms)"),
                    paste0("INT8 (Mean: ", round(mean(int8_runs), 1), " ms, p95: ", round(quantile(int8_runs, 0.95), 1), " ms)")),
         col = c(color_npu_w4, color_npu_int8), lty = 1, lwd = 2, bg = "white")
  dev.off()
  cat("[OK] Generated:", out_v4_png, "\n")
}

# ------------------------------------------------------------------------------
# 3. MobileNetV5 Native Stability Plot (100 Consecutive Passes on Hexagon NPU)
# ------------------------------------------------------------------------------
if (file.exists(v5_csv)) {
  df_v5 <- read.csv(v5_csv, stringsAsFactors = FALSE)
  df_v5$Latency_ms <- as.numeric(df_v5$Latency_ms)
  
  out_v5_png <- file.path(results_base, "MobileNetV5/machine_2/mobilenetv5_machine_2_native_stability.png")
  png(filename = out_v5_png, width = 1200, height = 600, res = 130)
  par(mar = c(5, 5, 4, 2), bg = "#FAFAFA")
  
  w4_runs <- df_v5$Latency_ms[df_v5$Format == "W4A8"]
  int8_runs <- df_v5$Latency_ms[df_v5$Format == "INT8"]
  
  plot(1:length(w4_runs), w4_runs, type = "l", col = color_npu_w4, lwd = 2,
       ylim = c(min(c(w4_runs, int8_runs)) * 0.95, max(c(w4_runs, int8_runs)) * 1.05),
       xlab = "Inference Run ID (100 Consecutive Field Images)", ylab = "Hardware Latency (ms)",
       main = "MobileNetV5: Native HTP Stability on Machine 2 (Hexagon NPU, 100 Runs)",
       las = 1, font.main = 2)
  grid(col = grid_color, lty = "dotted")
  lines(1:length(int8_runs), int8_runs, col = color_npu_int8, lwd = 2)
  
  legend("topright",
         legend = c(paste0("W4A8 (Mean: ", round(mean(w4_runs), 1), " ms, p95: ", round(quantile(w4_runs, 0.95), 1), " ms)"),
                    paste0("INT8 (Mean: ", round(mean(int8_runs), 1), " ms, p95: ", round(quantile(int8_runs, 0.95), 1), " ms)")),
         col = c(color_npu_w4, color_npu_int8), lty = 1, lwd = 2, bg = "white")
  dev.off()
  cat("[OK] Generated:", out_v5_png, "\n")
}

# ------------------------------------------------------------------------------
# 4. ScaleDetection V4 Native Plot (100 Consecutive Passes on Hexagon NPU)
# ------------------------------------------------------------------------------
if (file.exists(scale_csv)) {
  df_scale <- read.csv(scale_csv, stringsAsFactors = FALSE)
  df_scale$Latency_ms <- as.numeric(df_scale$Latency_ms)
  
  out_scale_png <- file.path(results_base, "ScaleDetection/machine_2/scaledetection_machine_2_native_stability.png")
  png(filename = out_scale_png, width = 1200, height = 600, res = 130)
  par(mar = c(5, 5, 4, 2), bg = "#FAFAFA")
  
  w4_runs <- df_scale$Latency_ms[df_scale$Format == "W4A8"]
  int8_runs <- df_scale$Latency_ms[df_scale$Format == "INT8"]
  
  plot(1:length(w4_runs), w4_runs, type = "l", col = color_npu_w4, lwd = 2,
       ylim = c(min(c(w4_runs, int8_runs)) * 0.9, max(c(w4_runs, int8_runs)) * 1.15),
       xlab = "Inference Run ID (100 Consecutive Macro Images)", ylab = "Hardware Latency (ms)",
       main = "ScaleDetection V4: 1024x1024 Metrology Stability on Machine 2 (Hexagon NPU, 100 Runs)",
       las = 1, font.main = 2)
  grid(col = grid_color, lty = "dotted")
  lines(1:length(int8_runs), int8_runs, col = color_npu_int8, lwd = 2)
  
  legend("topright",
         legend = c(paste0("W4A8 (Mean: ", round(mean(w4_runs), 1), " ms, p95: ", round(quantile(w4_runs, 0.95), 1), " ms)"),
                    paste0("INT8 (Mean: ", round(mean(int8_runs), 1), " ms, p95: ", round(quantile(int8_runs, 0.95), 1), " ms)")),
         col = c(color_npu_w4, color_npu_int8), lty = 1, lwd = 2, bg = "white")
  dev.off()
  cat("[OK] Generated:", out_scale_png, "\n")
}

cat("[SUCCESS] All 100-run native Snapdragon NPU plots generated successfully in English.\n")
