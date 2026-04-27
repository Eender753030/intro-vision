import argparse
import torch

from torchinfo import summary

from config import get_config
from model import get_model
from logger import get_logger
from trainer import Trainer
from tester import Tester
from inferencer import Inferencer

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
    
    if args.infer is not None:
        inferencer = Inferencer(model, device, args.infer, logger)
        inferencer.infer()
        
    elif args.test:
        tester = Tester(model, device, logger, config)
        tester.test()
        
    else:
        trainer = Trainer(model, device, logger, config)
        trainer.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Default is training mode")
    parser.add_argument("-t", "--test", action="store_true", help="Start with testing mode")
    parser.add_argument("-i", "--infer", type=str, default=None, help="Start with inference mode. Need input image path.")
    
    args = parser.parse_args()
    main(args)