import torch
from pathlib import Path

# Bootstrap paths
import sys
import os 
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from utils.paths import LOG_DIR, ROOT, MODEL_DIR
from dataloader import get_dataloader
from config import get_config
from logger import get_logger

# Import esp_ppq for quantization
try:
    from esp_ppq.api import espdl_quantize_onnx
except ImportError:
    print("Error: esp_ppq not found. Please run this in the pixi environment.")
    sys.exit(1)

DEVICE = "cpu"

def collate_fn(batch):
    batch = batch[0].to(DEVICE)
    return batch
    
def main():
    logger = get_logger("espdl_export", str(LOG_DIR))
    config = get_config()
    
    data_path = config["data"]["path"]
    if not os.path.isabs(data_path):
        data_path = str(ROOT / data_path)
        
    dataloader = get_dataloader(data_path, config, logger, training=False)
    
    onnx_path = str(MODEL_DIR / "model_simplified.onnx")
    espdl_path = str(MODEL_DIR / "model.espdl")
    
    ppq_graph = espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=espdl_path,
        calib_dataloader=dataloader,
        calib_steps=config["data"]["batch_size"],
        input_shape=[1, 1, 48, 48],
        inputs=None,
        target="esp32s3",
        num_of_bits=8,
        collate_fn=collate_fn,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )
    
    
if __name__ == "__main__":
    main()
