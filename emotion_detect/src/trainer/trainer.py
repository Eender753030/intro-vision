import os
import torch 
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo import summary
from tqdm import tqdm
from logging import Logger
from prettytable import PrettyTable

from dataloader import get_dataloader, get_classes_weight
from model import SimpleCNN
from model.loss import CenterLoss
import numpy as np

from utils.paths import MODEL_DIR, LOG_DIR


def mixup_data(x, y, alpha=0.2, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class EarlyStopping:
    """
    Manage when the training need to stop earlier.

    Args: 
        patience: How many epochs that validation loss not update will end the training.
        min_delta: The delta of loss change.
    """
    def __init__(self, patience: int = 7, min_delta: float = 1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.count = 0
        self.best_epoch = 1
        self.best_val_loss = float("inf")
        
    def update_and_check(self, val_loss: float, epoch: int) -> bool:
        """
        Update best_val_loss if the val_loss is better than the best in past, and check early stop or not.
        """
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
    """
    A class for training workflow.

    Args:
        model: The model that will be tested.
        device: The working processer, cuda or cpu.
        logger: The Logger to record information or warning.
        config: The configuration settings.
    """
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
        self.train_dataloder, self.valid_dataloader = get_dataloader(config["data"]["path"], config, logger, training=True)

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
        
        self.optimizer = SGD(
            model.parameters(), 
            lr=train_config["lr"],
            momentum=0.9,
            weight_decay=train_config["weight_decay"],
            nesterov=True
        )
        
        self.schedular = CosineAnnealingLR(
            self.optimizer, 
            T_max=train_config["epochs"],
            eta_min=1e-6
        )

        # Center Loss setup
        self.center_loss = CenterLoss(
            num_classes=config["model"]["num_class"],
            feat_dim=config["model"]["base_channels"] * 4, # Final channels after GAP
            use_gpu=(device.type == 'cuda')
        )
        self.optimizer_center = SGD(
            self.center_loss.parameters(),
            lr=config["train"]["loss"]["center_loss_lr"]
        )
        self.center_loss_weight = config["train"]["loss"]["center_loss_weight"]
        
        os.makedirs(MODEL_DIR, exist_ok=True)

        self.logger.info(f"Model summary:\n{summary(self.model, (1, 1, 48, 48), verbose=0)}")
        
    def train(self):
        """
        Start training. 
        """
        for epoch in range(1, self.epochs+1):
            self.model.train()
            
            epoch_loss = self._train_epoch(epoch)
            
            if self.schedular is not None:
                self.schedular.step()
                
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
                checkpoint_path = MODEL_DIR / "best_model.pt"
                torch.save(checkpoint, checkpoint_path)
                self.logger.info(f"Best model saved to {checkpoint_path}")
            
            if self.stopper is not None and self.stopper.update_and_check(epoch_val_loss, epoch):
                self.logger.info(f"Validation loss not improved. Early stop at epoch {epoch}.")
                break
            
        self._draw_result_plot()
        
    def _train_epoch(self, epoch: int) -> float:
        """
        Run an epoch of training.
        """
        running_loss = 0
        
        for images, labels in tqdm(self.train_dataloder, desc=f"Training Epoch:{epoch}/{self.epochs}:"):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Mixup Data
            alpha = self.config["train"].get("mixup_alpha", 0.0)
            if alpha > 0:
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha, self.device)
            else:
                targets_a, targets_b, lam = labels, labels, 1.0
            
            self.optimizer.zero_grad()
            self.optimizer_center.zero_grad()
            
            logits, features = self.model(images)
            
            # Mixed Loss Calculation
            l_ce = lam * self.loss(logits, targets_a) + (1 - lam) * self.loss(logits, targets_b)
            l_center = lam * self.center_loss(features, targets_a) + (1 - lam) * self.center_loss(features, targets_b)
            loss = l_ce + self.center_loss_weight * l_center
            
            loss.backward()
            
            self.optimizer.step()
            # Multiple centers by inverse of learning rate to keep them in scale? 
            # No, standard center loss optimizer handles it.
            for param in self.center_loss.parameters():
                param.grad.data *= (1. / self.center_loss_weight)
            self.optimizer_center.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / max(1, len(self.train_dataloder.dataset))
        return epoch_loss
    
    def _valie_epoch(self, epoch: int) -> tuple[float, float]:
        """
        Run an epoch of validation.
        """
        self.model.eval()
        running_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.valid_dataloader, desc=f"Valid Epoch: {epoch}/{self.epochs}"):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
            
                logits, _ = self.model(images)
                
                val_loss = self.loss(logits, labels)
                
                running_val_loss += val_loss.item() * images.size(0)
                
                probs = torch.softmax(logits, dim=1)
                _, predicted = torch.max(probs, dim=1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / max(1, len(self.valid_dataloader.dataset))
        
        epoch_accuracy = 100.0 * correct / max(1, total)
        
        return epoch_val_loss, epoch_accuracy
       
    def _draw_result_plot(self):
        """
        Draw the result plot and save.
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        plt.plot(self.metrics["loss"])
        plt.plot(self.metrics["val_loss"])
        
        plt.legend(["Train", "Valid"])
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        os.makedirs(LOG_DIR, exist_ok=True)
        plt.savefig(LOG_DIR / "training_result.png")
        plt.close()
        
        plt.figure(figsize=(12, 8))
        plt.plot(self.metrics["val_acc"], color="green")
        
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        
        plt.savefig(LOG_DIR / "validation_accuracy.png")
        plt.close() 
        