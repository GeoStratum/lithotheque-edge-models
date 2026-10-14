Sys.setenv(R_HOME = "E:/R/R/R-4.6.1")
csv_file <- "E:/fine tuning/rock identifier/lithotheque-edge-models/results/benchmark_workstation_pc_models_v4.csv"
output_dir <- "E:/fine tuning/rock identifier/lithotheque-edge-models/results"

df <- read.csv(csv_file, stringsAsFactors = FALSE)
df$Latency_num <- suppressWarnings(as.numeric(df$Latency_ms))
df_valid <- df[!is.na(df$Latency_num), ]

png_file <- file.path(output_dir, "8_mobilenetv4_workstation_rtx5060_comparison.png")
png(filename = png_file, width = 1100, height = 650, res = 130)
par(mar = c(5, 5, 4, 2), bg = "#FAFAFA")

means <- c(
  mean(df_valid$Latency_num[df_valid$Format == "W4A8" & df_valid$Backend == "GPU_RTX5060"]),
  mean(df_valid$Latency_num[df_valid$Format == "INT8" & df_valid$Backend == "GPU_RTX5060"]),
  mean(df_valid$Latency_num[df_valid$Format == "INT8" & df_valid$Backend == "CPU_i3_12100F"]),
  mean(df_valid$Latency_num[df_valid$Format == "W4A8" & df_valid$Backend == "CPU_i3_12100F"])
)

labels <- c("GPU W4A8\nRTX 5060", "GPU INT8\nRTX 5060", "CPU INT8\ni3-12100F", "CPU W4A8\ni3-12100F")
bar_colors <- c("#00b894", "#0984e3", "#d63031", "#e17055")

bp <- barplot(means, names.arg = labels, col = bar_colors, border = NA,
              ylim = c(0, max(means) * 1.3),
              main = "MobileNetV4: Workstation Latency (RTX 5060 Blackwell vs i3-12100F)",
              ylab = "Latence Moyenne (ms / image - 100 passes)",
              font.main = 2, col.lab = "#333333", col.axis = "#555555")
grid(col = "#E0E0E0", lty = "dotted")
barplot(means, names.arg = NA, col = bar_colors, border = NA, ylim = c(0, max(means) * 1.3), yaxt = "n", xaxt = "n", add = TRUE)

fps <- round(1000 / means, 1)
text(bp, means + 0.8, labels = paste0(round(means, 2), " ms\n(", fps, " fps)"), font = 2, cex = 0.9, col = "#222222")
abline(h = 16.67, col = "#636e72", lty = "dashed", lwd = 1.2)
text(x = 1.0, y = 18.0, labels = "Seuil 60 FPS (16.67 ms)", col = "#2d3436", cex = 0.85, font = 3)

legend("topright", 
       legend = c("GPU RTX 5060 Tensor Cores", "CPU i3-12100F x86 AVX2", "Seuil 60 FPS Temps Réel"),
       fill = c("#00b894", "#d63031", NA),
       border = c(NA, NA, NA),
       lty = c(NA, NA, 2), col = c(NA, NA, "#636e72"), lwd = c(NA, NA, 1.2),
       bg = "white", box.col = "#DDDDDD")
dev.off()
cat("Graphique généré avec succès :", png_file, "\n")
