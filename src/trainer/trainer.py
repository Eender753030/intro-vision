import os
import torch 
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from logging import Logger
from prettytable import PrettyTable

from dataloader import get_dataloader, get_classes_weight, get_mean_and_std
from model import SimpleCNN


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "model")


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.count = 0
        self.best_epoch = 1
        self.best_val_loss = float("inf")
        
    def update_and_check(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.count = 0      
        else:
            self.count += 1
            
        if self.count > self.patience:
            return True
        return False

class Trainer:
    def __init__(
        self,
        model: SimpleCNN,
        device: torch.device,
        logger: Logger,
        config: dict,
    ):
        train_config = config["train"]
        self.model = model
        self.device = device
        self.train_dataloder, self.valid_dataloader = get_dataloader(config["data"]["path"], config, training=True)

        self.logger = logger
        self.epochs = train_config["epochs"]
        self.config = config
        self.metrics = {
            "loss": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.stopper = EarlyStopping(
            train_config["early_stop"]["patience"], train_config["early_stop"]["min_delta"]
        ) if train_config["early_stop"]["enable"] else None
        
        self.best_val_loss = float("inf")
        
        self.loss = CrossEntropyLoss(
            weight=get_classes_weight(config["data"]["path"]).to(device),
            label_smoothing=config["train"]["label_smoothing"]
        )
        
        self.optimizer = Adam(
            model.parameters(), 
            lr=train_config["lr"],
            weight_decay=train_config["weight_decay"],
        )
        
        self.schedular = OneCycleLR(
            self.optimizer, 
            max_lr=train_config["max_lr"],
            epochs=train_config["epochs"],
            steps_per_epoch=len(self.train_dataloder)
        )
        
        os.makedirs(MODEL_PATH, exist_ok=True)
        
    def train(self):
        
        for epoch in range(1, self.epochs+1):
            self.model.train()
            
            epoch_loss = self._train_epoch(epoch)
                
            epoch_val_loss, epoch_accuracy = self._valie_epoch(epoch)
                                     
            table = PrettyTable(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])
            
            table.add_row([f"{epoch}/{self.epochs}", f"{epoch_loss:.4f}", f"{epoch_val_loss:.4f}", f"{epoch_accuracy:.2f}%"])
            
            self.logger.info(f"\n{table}")
            
            self.metrics["loss"].append(epoch_loss)
            self.metrics["val_loss"].append(epoch_val_loss)
            self.metrics["val_acc"].append(epoch_accuracy)
            
            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                checkpoint = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_val_loss": self.best_val_loss,
                    "epoch": epoch,
                    "config": self.config,
                }        
                torch.save(checkpoint, os.path.join(MODEL_PATH, "best_model.pt"))
                self.logger.info(f"Save best model at epoch {epoch}.")
            
            if self.stopper is not None and self.stopper.update_and_check(epoch_val_loss, epoch):
                self.logger.info(f"Validation loss not improved. Early stop at epoch {epoch}.")
                break
            
        self._draw_result_plot()
        
    def _train_epoch(self, epoch: int) -> float:
        running_loss = 0
        
        
        for images, labels in tqdm(self.train_dataloder, desc=f"Training Epoch:{epoch}/{self.epochs}:"):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            
            loss = self.loss(outputs, labels)
            
            loss.backward()
            
            self.optimizer.step()
            
            if self.schedular is not None:
                self.schedular.step()
                
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / max(1, len(self.train_dataloder.dataset))
        return epoch_loss
    
    def _valie_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.eval()
        running_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.valid_dataloader, desc=f"Valid Epoch: {epoch}/{self.epochs}"):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
            
                outputs = self.model(images)
                
                val_loss = self.loss(outputs, labels)
                
                running_val_loss += val_loss.item() * images.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, dim=1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / max(1, len(self.valid_dataloader.dataset))
        
        epoch_accuracy = 100.0 * correct / max(1, total)
        
        return epoch_val_loss, epoch_accuracy
       
    def _draw_result_plot(self):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        plt.plot(self.metrics["loss"])
        plt.plot(self.metrics["val_loss"])
        
        plt.legend(["Train", "Valid"])
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        plt.savefig("log/training_result.png")
        plt.close()
        
        plt.figure(figsize=(12, 8))
        plt.plot(self.metrics["val_acc"], color="green")
        plt
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        
        plt.savefig("log/validation_accuracy.png")
        plt.close() 
        