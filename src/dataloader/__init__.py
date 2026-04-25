import os
import torch
import json
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

from .dataloader import RafDbDataset
from .preprocces import *

def get_dataloader(data_path: os.PathLike, config: dict, training: bool = True) -> tuple[DataLoader, DataLoader] | DataLoader: 
    try:
        with open("model/normalize_data.json", "r") as f:
            d = json.load(f)
        mean = d["mean"]
        std = d["std"]
    except:
        if not training:
            raise RuntimeError("No normalize_data found. Try training first.")
        mean, std = get_mean_and_std(data_path)
        d = {"mean": mean, "std": std}
        with open("model/normalize_data.json", "w") as f:
            json.dump(d, f)

    if training:
        transform = v2.Compose([
            v2.Grayscale(),
            v2.Resize((48, 48)),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation(15),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[mean], std=[std]),
        ])
    else:
        transform = v2.Compose([
            v2.Grayscale(),
            v2.Resize((48, 48)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[mean], std=[std]),
        ])
        
    dataset = RafDbDataset(data_path, training=training, transform=transform)
    g = torch.Generator().manual_seed(42)
    
    if not training:
        return DataLoader(
            dataset,
            batch_size=1, 
            shuffle=False,
            generator=g,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )
    
    len_dataset = len(dataset)
    train_size = int(0.70 * len_dataset)
    val_size = len_dataset - train_size
    
    train, valid = random_split(dataset, [train_size, val_size], generator=g)

    train_loader = DataLoader(
        train, 
        batch_size=config["data"]["batch_size"], 
        shuffle=True,
        generator=g,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid, 
        batch_size=config["data"]["batch_size"], 
        shuffle=False,
        generator=g,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    
    return train_loader, valid_loader