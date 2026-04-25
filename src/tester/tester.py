import torch
import time
from logging import Logger
from tqdm import tqdm

from model import SimpleCNN
from dataloader import get_dataloader

class Tester:
    def __init__(
        self, 
        model: SimpleCNN,
        device: torch.device,
        logger: Logger,
        config: dict, 
    ):
        self.model = model
        checkpoint = torch.load("model/best_model.pt", weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device)
        self.model.eval()
        
        self.device = device
        self.logger = logger
        self.config = checkpoint["config"]
        
        self.dataloader = get_dataloader(config["data"]["path"], config, training=False)
        
    def test(self):
        total = 0
        correct = 0
        total_time = 0
        dummy_input = torch.randn(1, 1, 48, 48).to(self.device)
        self.logger.info("Warnup first...")
        with torch.no_grad():
            for _ in tqdm(range(100), desc="Warmup"):
                _ = self.model(dummy_input)
        
        self.logger.info("Ready!")
        
        self.logger.info("Start testing...")
        
        with torch.no_grad():
            for image, label in tqdm(self.dataloader, desc="Testing"):
                image = image.to(self.device)
                label = label.to(self.device)
    
                start = time.perf_counter()
                output = self.model(image)
                end = time.perf_counter()
                
                _, predicted = torch.max(output, dim=1)

                total_time += end - start
                total += 1
                correct += 1 if predicted == label else 0
                
        accuracy = correct / total * 100
        
        self.logger.info(f"Test accuracy: {accuracy:.2f}%. Process data count: {total}. Cost time: {total_time:.2f}s. Avg time: {total_time / total * 1000:.3f}ms")
                 
                
        