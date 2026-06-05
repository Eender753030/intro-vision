# ESP32-S3 Emotion Detection Model Comparison

---

## 📊 Table 1: Model Accuracy, Benchmark Performance & Latency

This table compares model accuracy, parameters, MACs (computation), raw performance benchmarked on the ESP32-S3 (warmup: 2, test: 10), and the **integrated inference time** (which includes image pre-processing, cropping, and grayscale conversion overhead).

| Candidate Model Name | Params | MACs (Mult-Adds) | PyTorch Acc | ESPDL Sim Acc (INT8) | Raw Benchmark Latency | Est. Throughput | Integrated Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **🥇 simple_cnn_2stage_16channels** | **53,815** (Min) | 18.91 M | 71.61% | **71.81%** | **30.04 ms** (Fastest) | **33.29 FPS** (Highest) | **30 ms ~ 60 ms** (Best) |
| **simple_cnn_3stage_8channels** | 64,847 | **8.13 M** (Lowest) | 71.06% | 71.19% | 47.33 ms | 21.13 FPS | 70 ms ~ 80 ms |
| **simple_cnn_4stage_8channels** | 268,591 | 11.45 M | **74.22%** (Highest)| **74.22%** (Highest) | 69.47 ms | 14.39 FPS | 120 ms ~ 170 ms |
| **simple_cnn_3stage_16channels** | 257,559 | 32.18 M | 73.24% | 73.53% | 151.04 ms | 6.62 FPS | 350 ms ~ 550 ms |
| **simple_dwcnn_3stage_16channels** | 56,167 | 9.42 M | 65.74% | 65.87% | 69.47 ms | 14.39 FPS | 120 ms ~ 160 ms |
| **simple_dwcnn_3stage_32channels** | 210,119 | 35.36 M | 73.66% | 73.37% | 374.89 ms | 2.67 FPS | 470 ms ~ 540 ms |
| **simple_dwcnn_4stage_16channels** | 220,839 | 13.34 M | 70.01% | 69.95% | 108.69 ms | 9.20 FPS | 170 ms ~ 230 ms |

---

## 💾 Table 2: Memory Footprint on ESP32-S3

This table displays the static and peak SRAM allocations (which reside in the ESP32-S3's internal high-speed memory) as well as the external PSRAM footprint utilized during model initialization.

| Candidate Model Name | Static SRAM Used | Peak SRAM Used (Min Ever) | PSRAM Used |
| :--- | :---: | :---: | :---: |
| **🥇 simple_cnn_2stage_16channels** | **14.2 KB** (Lowest) | **26.6 KB** (Lowest) | 136.8 KB |
| **simple_cnn_3stage_8channels** | 19.8 KB | 39.9 KB | **117.8 KB** (Lowest) |
| **simple_cnn_4stage_8channels** | 24.6 KB | 51.3 KB | 322.9 KB |
| **simple_cnn_3stage_16channels** | 19.8 KB | 39.9 KB | 342.5 KB |
| **simple_dwcnn_3stage_16channels** | 24.6 KB | 51.3 KB | 322.9 KB |
| **simple_dwcnn_3stage_32channels** | 29.1 KB | 62.1 KB | 381.6 KB |
| **simple_dwcnn_4stage_16channels** | 39.3 KB | 86.2 KB | 331.0 KB |

---

## 🎯 Final Deployed Model: `simple_cnn_2stage_16channels`

- **Best Real-time Responsiveness**: Its integrated inference time is **30 ms ~ 60 ms** (equivalent to ~17-33 FPS in production), which makes the debug UI animations and haptic transitions extremely snappy and smooth.
- **Resource Safe**: Only consumes **14.2 KB of SRAM** and **136.8 KB of PSRAM**, leaving plenty of headroom for FreeRTOS tasks, BLE/Wi-Fi operations, and camera frame grabbers without risking memory exhaustion crashes.
- **High Efficiency**: Achieving **71.81% INT8 accuracy** while running significantly faster and using less memory compared to more complex options.
