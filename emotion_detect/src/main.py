import argparse
import torch
import os

from torchinfo import summary

from config import get_config
from model import get_model
from logger import get_logger
from trainer import Trainer
from tester import Tester


def main(args: argparse.Namespace):
    logger = get_logger("intro-vision", "log")
    
    config = get_config()
    logger.info(f"Loading config successed")
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = get_model(config)
    model = model.to(device)
    logger.info("Model build completed")
    logger.debug('\n' + str(summary(model, (1, 1, 48, 48), verbose=0)))
    
    if args.test:
        tester = Tester(model, device, logger, config)
        tester.test()
        
    else:
        trainer = Trainer(model, device, logger, config)
        trainer.train()
    

if __name__ == "__main__":
    # Lock the working directory to the root of emotion_detect/
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))   

    parser = argparse.ArgumentParser(description="Default is training mode")
    parser.add_argument("--test", action="store_true", help="Start with testing mode")
    
    args = parser.parse_args()
    main(args)
