# Intro-Vision
Still working on...

---

## Environment Setup

This project uses [Pixi](https://pixi.sh/) for cross-platform package and environment management, ensuring identical dependency tracking across systems.

### 1. Install Pixi
```bash
# Download and install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Refresh shell environment variables
source ~/.bashrc

# Verify the installation
pixi --version
```

### 2. Clone Submodules
The firmware depends on the custom `esp-who` components. Fetch the submodules immediately after cloning the repository:
```bash
git submodule update --init --recursive
```

### 3. Install Dependencies
Run the following inside the project root to configure and download both Python packages and the ESP-IDF toolchain bindings:
```bash
pixi install
```

---

## Commands Workflow

Pixi wraps the entire workflow—from training down to ESP32 flashing—into simple CLI tasks. You can view all configurations inside [pixi.toml](file:///home/eender/Workspace/Project/intro-vision/pixi.toml).

### Phase 1: Model Training & Inference (PyTorch)
Configure hyperparameters inside [config.toml](file:///home/eender/Workspace/Project/intro-vision/emotion_detect/config.toml).

*   **Train Model**:
    ```bash
    pixi run emo_train
    ```
*   **Test Model** (Validate metrics):
    ```bash
    pixi run emo_test
    ```
*   **Inference Image** (Test local predictions):
    ```bash
    pixi run emo_infer <path_to_image>
    ```

### Phase 2: ONNX Export & ESP-DL Quantization
Transition your PyTorch weights (`.pt`) into optimized edge deployment formats (`.espdl`).

*   **Export to ONNX**:
    ```bash
    pixi run emo_onnx_export
    ```
*   **Simplify ONNX Model** (Optimizes layers for quantization):
    ```bash
    pixi run emo_onnx_sim
    ```
*   **Verify ONNX Model**:
    ```bash
    pixi run emo_onnx_test
    ```
*   **Quantize & Export to ESP-DL** (Generates `INT8` model):
    ```bash
    pixi run emo_espdl_export
    ```
*   **Simulate ESP-DL Accuracy**:
    ```bash
    pixi run emo_espdl_test
    ```

### Phase 3: ESP32-S3 Firmware Deployment
Once you have the `.espdl` model, convert it into C++ arrays and compile the ESP-IDF firmware.

*   **Generate Model Header**:
    ```bash
    pixi run espdl_to_firmware
    ```
*   **Initialize target as ESP32-S3**:
    ```bash
    pixi run esp_init
    ```
*   **Build Firmware**:
    ```bash
    pixi run esp_build
    ```
*   **Flash Firmware**:
    ```bash
    pixi run esp_flash
    ```
*   **Flash and Monitor Serial**:
    ```bash
    pixi run esp_flash_m
    ```
*   **Monitor Serial**:
    ```bash
    pixi run esp_monitor
    ```
*   **Configure Build/Peripherals**:
    ```bash
    pixi run esp_menuconfig
    ```
