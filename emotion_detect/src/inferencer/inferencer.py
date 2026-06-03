import os
import json
import torch
import torchvision
from torchvision.transforms import v2
from logging import Logger
from pathlib import Path
from utils.paths import MODEL_DIR

from model import SimpleCNN


class Inferencer:   
    def __init__(
        self,
        model: SimpleCNN,
        device: torch.device,
        input_path: os.PathLike,
        logger: Logger,
    ):
        self.logger = logger
        self.logger.info("Loading state_dict into model...")
        checkpoint = torch.load(MODEL_DIR / "best_model.pt", weights_only=True)
        
        self.model = model
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device)
        self.model.eval()

        self.logger.info("Loading completed")

        self.config = checkpoint["config"]
        
        
        self.emotion_table = {
            0: "Suprise",
            1: "Fear",
            2: "Disgust",
            3: "Happiness",
            4: "Sadness",
            5: "Anger",
            6: "Neutral",
        }
        
        self.device = device
        self.input_path = input_path 
        
    def infer(self):
        self.logger.info("Start inferencing...")
        img = self._load_image().to(self.device)
        
        with torch.no_grad():
            output, _ = self.model(img)
            
            prob = torch.nn.functional.softmax(output, dim=1).squeeze()

            _, predicted = torch.max(prob, dim=0)

            emotion = self.emotion_table[predicted.item()]
            
        self.logger.info("Inferencing completed.")
  
        prob_str = ""
        for i in range(7):
            prob_str += f"{self.emotion_table[i]}: {prob[i].item() * 100:.2f}%. "
        
        self.logger.info(f"Emotion reuslt: {emotion}. Probability for each: {prob_str}")

        self._draw_result(img, emotion)
        
    def _load_image(self) -> torch.Tensor:       
        from torchvision.io import ImageReadMode
        img = torchvision.io.read_image(str(self.input_path), mode=ImageReadMode.RGB)
        img = img.unsqueeze(0)
        norm_json_path = MODEL_DIR / "normalize_data.json"
        with open(norm_json_path, 'r') as f:
            j = json.load(f)
            mean, std = j["mean"], j["std"]
            
        transform = v2.Compose([
            v2.Grayscale(),
            v2.Resize((48, 48)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([mean], [std]),
        ])
        
        return transform(img)    
        
    def _draw_result(self, image: torch.Tensor, result: str):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 8))
        
        plt.imshow(image.squeeze().detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"Your image's emotion is '{result}'")
        
        plt.show()
