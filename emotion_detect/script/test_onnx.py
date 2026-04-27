import sys
import onnxruntime as ort
import numpy as np
import time
from tqdm import tqdm

sys.path.append("src")

from dataloader import get_dataloader
from config import get_config
from logger import get_logger


def main():
    logger = get_logger("onnx_test", "log")
    
    logger.info("Building ONNX model...")
    session = ort.InferenceSession("model/model_simplified.onnx",  providers=['CUDAExecutionProvider'])
    logger.info("Build completed")
    
    config = get_config()
    loader = get_dataloader("archive", config, training=False)
    
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
    