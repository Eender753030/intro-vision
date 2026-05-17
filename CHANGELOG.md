# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
