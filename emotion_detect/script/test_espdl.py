import time
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

# Bootstrap paths
import sys
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from utils.paths import LOG_DIR, ROOT
from dataloader import get_dataloader
from config import get_config
from logger import get_logger

# Import esp_ppq components
try:
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.executor.torch import TorchExecutor
except ImportError:
    print("Error: esp_ppq components not found. Please run this in the pixi environment.")
    sys.exit(1)

def main():
    onnx_path = str(ROOT / "model" / "model_simplified.onnx")
    logger = get_logger("espdl_test", str(LOG_DIR))

    if not os.path.exists(onnx_path):
        logger.error(f"ONNX model not found at {onnx_path}. Need ONNX to simulate quantization.")
        return

    # 1. Load Config and Dataloader
    config = get_config()
    data_path = config["data"]["path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(base_dir, data_path)

    loader = get_dataloader(data_path, config, logger, training=False)
    
    # 2. Simulate Quantization Inference
    # Use device from config
    device = config.get("device", "cpu").lower()
    logger.info(f"Initializing Quantization Simulation on {device}...")
    
    # Wrapper to only yield images for calibration
    class CalibWrapper:
        def __init__(self, loader, device):
            self.loader = loader
            self.device = device
        def __iter__(self):
            for img, _ in self.loader:
                yield img.to(self.device)
        def __len__(self):
            return len(self.loader)

    # This will return a quantized graph
    graph = espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=os.path.join(base_dir, "model", "tmp.espdl"),
        calib_dataloader=CalibWrapper(loader, device),
        calib_steps=8,
        input_shape=[1, 1, 48, 48],
        device=device
    )
    
    # Create executor from graph for simulation
    executor = TorchExecutor(graph)
    
    logger.info("Starting accuracy test (INT8 Simulation)...")
    
    total = 0
    correct = 0
    start_time = time.time()
    
    # 3. Iterate through data
    for img, label in tqdm(loader, desc="Testing INT8 Sim"):
        # Run simulated inference
        img = img.to(device)
        outputs = executor.forward(img)
        
        output = outputs[0]
        if torch.is_tensor(output):
            output_np = output.detach().cpu().numpy()
        else:
            output_np = output
            
        # Calculate accuracy
        preds = np.argmax(output_np, axis=1)
        actuals = label.numpy()
        
        correct += np.sum(preds == actuals)
        total += len(actuals)

    end_time = time.time()
    cost_time = end_time - start_time
    accuracy = correct / total * 100
    
    logger.info(f"Test Completed.")
    logger.info(f"Total Images: {total}")
    logger.info(f"Correct: {correct}")
    logger.info(f"Simulated INT8 Accuracy: {accuracy:.2f}%")
    logger.info(f"Total Cost Time: {cost_time:.2f}s")
    logger.info(f"Avg Inference Time: {cost_time/total*1000:.3f}ms per image")

if __name__ == "__main__":
    main()
