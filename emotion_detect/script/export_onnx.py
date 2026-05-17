import torch
from pathlib import Path

# Bootstrap paths
import sys
import os
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.append(str(_src_dir))

from utils.paths import LOG_DIR, ROOT
from model import get_model
from logger import get_logger

def main():
    model_dir = ROOT / "model"
    logger = get_logger("onnx_export", str(LOG_DIR))
 
    logger.info("Building PyTorch model...")
    
    checkpoint_path = model_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model = get_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    device = torch.device("cpu")
    model.to(device)
 
    logger.info("Build completed")
    
    dummy_input = torch.randn(1, 1, 48, 48).to(device)
    
    onnx_path = os.path.join(model_dir, "model.onnx")
    logger.info(f"Exporting ONNX to {onnx_path}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            f=onnx_path,
            export_params=True,
            input_names=["image"],
            output_names=["emotion"],
            opset_version=12,
        )
    logger.info(f"Export completed.")



if __name__ == "__main__":
    main()