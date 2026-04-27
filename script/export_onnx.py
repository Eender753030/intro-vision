import torch

import sys

sys.path.append("src")

from model import get_model
from logger import get_logger

def main():
    logger = get_logger("onnx_exporter", "log")
 
    logger.info("Building PyTorch model...")
    
    checkpoint = torch.load("model/best_model.pt", weights_only=True)
    model = get_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    device = torch.device("cpu")
    model.to(device)

    logger.info("Build completed")
    
    dummy_input = torch.randn(1, 1, 48, 48).to(device)
    
    logger.info(f"Exporting ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            f="model/model.onnx",
            export_params=True,
            input_names=["image"],
            output_names=["emotion"],
            opset_version=12,
            # dynamo=True,
        )
    logger.info(f"Export completed. Model store at model/model.onnx")



if __name__ == "__main__":
    main()