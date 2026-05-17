import os
import sys
import time
import serial
import re
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Bootstrap paths to find the 'src' directory
_script_dir = Path(__file__).resolve().parent
_root_dir = _script_dir.parent
_src_dir = _root_dir / "src"

if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from utils.paths import LOG_DIR
from dataloader import get_dataloader
from config import get_config
from logger import get_logger

def main():
    logger = get_logger("hardware_test", str(LOG_DIR))

    # 1. Serial Configuration
    # Adjust port as needed (e.g., /dev/ttyUSB0 or COM3)
    SERIAL_PORT = "/dev/ttyACM0" 
    BAUD_RATE = 115200
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        logger.info(f"Connected to ESP32 on {SERIAL_PORT}")
    except Exception as e:
        logger.error(f"Failed to connect to Serial: {e}")
        return

    # 2. Load Dataloader
    config = get_config()
    data_path = config["data"]["path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(base_dir, data_path)

    loader = get_dataloader(data_path, config, logger, training=False)
    
    total = 0
    correct = 0
    
    EMOTIONS = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

    logger.info("Starting Hardware-in-the-Loop Accuracy Test...")
    
    # Wait for ESP32 to be ready
    time.sleep(2)
    ser.reset_input_buffer()

    start_time_all = time.time()
    latencies = []

    # Get mean/std for de-normalization if needed
    MEAN = config["data"].get("mean", 0.4793299436569214)
    STD = config["data"].get("std", 0.23705872893333435)

    for img, label in tqdm(loader, desc="Testing on ESP32"):
        # 1. Wait for ESP32 Ready Signal
        ready = False
        while not ready:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "WAITING_FOR_IMAGE" in line:
                ser.reset_input_buffer()
                ready = True
        
        # 2. De-normalize to [0, 1] then to [0, 255]
        # img is [(img - mean) / std] -> we want [img * 255]
        # If your dataloader already has Normalize, we must reverse it
        img_raw = (img[0, 0].numpy() * STD + MEAN) * 255.0
        img_uint8 = np.clip(img_raw, 0, 255).astype(np.uint8)
        raw_bytes = img_uint8.tobytes()
        
        image_start = time.time()
        # Send to ESP32
        ser.write(raw_bytes)
        ser.flush()
        
        # Wait for result and monitor progress
        result_found = False
        timeout_start = time.time()
        while not result_found and (time.time() - timeout_start < 10):
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            # if "PROGRESS" in line:
            #    print(f"  {line}") # Optional: debug progress
            if "Final Decision:" in line:
                latencies.append(time.time() - image_start)
                
                # Extract emotion name
                match = re.search(r"Final Decision: (\w+)", line)
                if match:
                    pred_emotion = match.group(1)
                    actual_emotion = EMOTIONS[label[0].item()]
                    
                    if pred_emotion == actual_emotion:
                        correct += 1
                    total += 1
                    result_found = True
            elif "DONE" in line:
                break

    end_time_all = time.time()

    if total > 0:
        total_duration = end_time_all - start_time_all
        avg_latency = (sum(latencies) / len(latencies)) * 1000 # in ms
        
        accuracy = correct / total * 100
        logger.info(f"Hardware Test Completed.")
        logger.info(f"Total Images: {total}")
        logger.info(f"Correct: {correct}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"Total Time: {total_duration:.2f}s")
        logger.info(f"Avg Latency (Comm + Infer): {avg_latency:.1f} ms per image")
    else:
        logger.error("No results received from hardware.")

    ser.close()

if __name__ == "__main__":
    main()
