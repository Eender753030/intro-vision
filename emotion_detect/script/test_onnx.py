import numpy as np
import os
import sys
import onnxruntime as ort
import time
from tqdm import tqdm
from pathlib import Path

# Bootstrap paths
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from utils.paths import LOG_DIR, ROOT, MODEL_DIR
from dataloader import get_dataloader
from config import get_config
from logger import get_logger

def main():
    model_path = str(MODEL_DIR / "model_simplified.onnx")
    logger = get_logger("onnx_test", str(LOG_DIR))
    
    config = get_config()
    providers = ['CUDAExecutionProvider'] if config["device"].lower() == "cuda" else ['CPUExecutionProvider']
    logger.info("Building ONNX model...")
    session = ort.InferenceSession(model_path, providers=providers)
    logger.info("Build completed")
    
    data_path = config["data"]["path"]
    if not os.path.isabs(data_path):
        data_path = str(ROOT / data_path)

    loader = get_dataloader(data_path, config, logger, training=False)
    
    dummy_input = {"image": np.random.randn(1, 1, 48, 48).astype(np.float32)}
    logger.info("Warmup first...")
    for _ in tqdm(range(100), desc="Warmup"):
        session.run(None, dummy_input)
        
    total = 0
    correct = 0
    total_time = 0.0
    
    for img, label in tqdm(loader, desc="Testing"):
        img = img.detach().cpu().numpy().astype(np.float32)
        label = label.detach().cpu().numpy().astype(np.float32)
        input = {"image": img}

        start = time.perf_counter()
        output = session.run(["emotion"], input)
        end = time.perf_counter()
        
        predict = np.argmax(output)
        
        total += 1
        correct += 1 if predict == label else 0
        total_time += end - start
        
    accuracy = correct / total * 100
    avg_time = total_time / total
    
    logger.info(f"Test accuracy: {accuracy:.2f}%. Process data count: {total}. Cost time: {total_time:.2f}s. Avg time: {avg_time * 1000:.3f}ms")

if __name__ == "__main__":
    main()
    