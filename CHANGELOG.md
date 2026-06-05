# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] - 2026-06-05

### Added
- **Happiness Haptic Feedback**: Added gentle double pulse vibration pattern for Happiness (`EmotionType::HAPPINESS`) in the firmware and mapping description in the UI.
- **Inference Benchmarking Program**: Implemented a standalone C++ performance testing suite (`model_performace.cpp`) on the ESP32 to benchmark model latencies.

### Changed
- **Depthwise Separable Convolution**: Added support for training and exporting Depthwise Separable CNN models, and migrated model activations to `ReLU6` to optimize edge compute.
- **LeakyReLU Activations**: Swapped standard `PReLU` with `LeakyReLU` in traditional CNN modules to improve inference times on the ESP32.
- **Base64 Serial Protocol**: Migrated debug image transmission from HEX to Base64, reducing serial overhead by ~33% and increasing UI frame rate.
- **Optimized Baud Rate & JPEG FPS**: Increased ESP32 console baud rate to 921600 bps and throttled JPEG transmission when a face is detected (once every 2 frames) to maximize system responsiveness and prevent UART bottlenecks.
- **Refactored Project Architecture**: Restructured the Python training codebase, separating execution logic into modular `runners/` (trainer and tester) and clean modules.
- **Configurable Center Loss**: Refactored the training pipeline to allow turning Center Loss on/off via configuration.

---

## [0.3.0] - 2026-06-03

### Added
- **Debug Dashboard**: Implemented a localhost real-time debug UI (`debug_dashboard/`) with HTML/JS/CSS to visualize model predictions, performance, and hardware states.
- **Haptic Feedback**: Developed DRV2605L haptic motor driver (`MotorDriver` class) mapping distinct vibration patterns to all 7 emotions.
- **Performance Profiling**: Created a C++ performance profiling helper (`PerformanceProfiler`) in the ESP32 firmware to track inference and preprocessing latencies.
- **Firmware Emotion Type**: Defined a clean `EmotionType` enum class representing the 7 emotion categories.
- **Camera & Haptics Pin Configuration**: Documented hardware pin mappings for OV2640 and DRV2605L with ESP32-S3, integrating them into `main.cpp`.
- **New Pixi Tasks**: Added `debug_ui` (starts local server for debug dashboard) and `esp_install` (installs ESP-IDF dependencies) to `pixi.toml`.

### Changed
- **Bilinear Interpolation Preprocessing**: Replaced old crop/resize logic with bilinear interpolation supporting `RGB565` format directly in firmware.
- **Firmware Code Refactoring**: Reorganized model deployment code layout, refactored `EmotionModel` integration, and cleaned up unused testing arrays and configurations.
- **Model Array Alignment**: Optimized the `espdl_to_firmware` script to include `__attribute__((aligned(16))) const` for ESP-DL data alignment.

### Fixed
- **Path and Export Issues**: Fixed PyTorch-to-ESP-DL pipeline path parsing and model export error script paths.

---

## [0.2.0] - 2026-05-17

### Added
- **Mixup Augmentation**: Added Mixup data augmentation to training pipeline (`mixup_alpha = 0.2`) to address dataset generalization limitations.
- **Center Loss**: Implemented `CenterLoss` in `src/model/loss.py` to cluster feature spaces and increase intra-class compactness.
- **Improved CNN Architecture**: 
  - Replaced standard `ReLU` with `PReLU` in `ConvBlock` and `ResBlock`.
  - Added dynamic `base_channels` configurations supporting scaled width (32 to 64).
  - Explicitly divided model into `stem`, `stages`, and `fc` structures for cleaner modularity.
- **Hardware Integration & Diagnostics**:
  - Implemented Python-to-ESP32 serial test suite (`test_esp32_hardware.py`) allowing simulated test set execution.
  - Implemented C++ simulation validation pipeline for ESP-DL models.
  - Managed centralized paths globally via `paths.py` utilities.
- **Submodule Integration**: Integrated `esp-who` fork as a submodule to support camera processing and model packaging.

### Changed
- **Pipeline Refactoring**: Split image transforms between training (random augmentations) and validation (deterministic resize & normalize), eliminating critical data pipeline leakage.
- **Optimization Strategy**: Migrated the training pipeline from `Adam` to `SGD` (`momentum=0.9`, `nesterov=True`) with a `CosineAnnealingLR` scheduler for smoother, deep convergence.
- **Log Metrics**: Enhanced logging features to trace double-loss values and validation performance visually.

### Fixed
- **Inference Issues**: Resolved inference script path parsing errors.
- **AttributeErrors**: Fixed missing architectural attributes (`stem`, `pool`, `stages`) in custom `SimpleCNN` initialization.

---

## [0.1.0] - 2026-04-28

### Added
- **ONNX Pipeline**: Supported exporting PyTorch training checkpoints directly to standard ONNX.
- **ESP-DL Deployment**: Developed toolchain scripts to quantize ONNX models into `INT8` format for ESP32-S3 execution.
- **Model Inference**: Implemented localized testing and standalone inference components for the PyTorch face emotion models.
- **Dataloader Automation**: Automated data preparation and automatic remote archive downloading.
- **Task Scripts**: Populated `pixi.toml` with automated CLI tasks (`emo_train`, `emo_test`, `emo_infer`, etc.).

### Changed
- **Architectural Cleanup**: Restructured code modularity from a flat script into dedicated `model/`, `dataloader/`, `trainer/`, and `tester/` packages.

### Fixed
- Corrected missing logger argument pass-through in emotion detection training setups.
