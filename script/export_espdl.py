import sys
from esp_ppq.api import espdl_quantize_onnx

sys.path.append("src")

from dataloader import get_dataloader
from logger import get_logger
from config import get_config

DEVICE = "cpu"

def collate_fn(batch):
    batch = batch[0].to(DEVICE)
    return batch
    
def main():
    logger = get_logger("espdl_exporter", "log")
    
    config = get_config()
    
    _, dataloader = get_dataloader(config["data"]["path"], config, training=True)
    
    ppq_graph = espdl_quantize_onnx(
        onnx_import_file="model/model.onnx",
        espdl_export_file="model/model.espdl",
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