import argparse
import torch
import os
from pathlib import Path

from config import get_config
from model import get_model
from logger import get_logger
from runners import Trainer, Tester, Inferencer
from utils.paths import ROOT, add_src_to_path

add_src_to_path()


def main(args: argparse.Namespace):
    config = get_config()
    logger = get_logger("intro-vision", str(ROOT / "log"))
    
    if args.infer is not None:
        input_path = Path(args.infer).resolve()
        if not input_path.exists():
            input_path = (ROOT / args.infer).resolve()
            
        logger.info(f"Inferencing image: {input_path}")
        if not input_path.exists():
            logger.error(f"Image not found at {input_path}")
            return
            
        device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        model = get_model(config).to(device)
        inferencer = Inferencer(model, device, input_path, logger)
        inferencer.infer()
        return

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = get_model(config)
    model = model.to(device)
    logger.info("Model build completed")
    
    if args.infer is not None:
        base_dir = os.getcwd()
        img_path = os.path.join(base_dir, args.infer)
    if args.test:
        tester = Tester(model, device, logger, config)
        tester.test()
        return

    # 4. Training Mode
    trainer = Trainer(model, device, logger, config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Detection Training and Inference")
    parser.add_argument("-t", "--test", action="store_true", help="Run testing mode")
    parser.add_argument("-i", "--infer", type=str, help="Run inference on an image")
    args = parser.parse_args()
    main(args)
