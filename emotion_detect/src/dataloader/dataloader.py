import os
import torch
import json
from logging import Logger
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

from .dataset import RafDbDataset
from .preprocess import get_mean_and_std


def get_dataloader(
    data_path: os.PathLike, 
    config: dict, 
    logger: Logger,
    training: bool = True
) -> tuple[DataLoader, DataLoader] | DataLoader:
    """
    Build the dataloader with RAF-DB dataset in training or testing.
    Crate transforms for pre-process of the images.
    
    Args:
        data_path: The path of folder where dataset is.
        config: The configuration settings.
        logger: The Logger to record information or warning.
        training: Is training or testing mode.
    
    Returns:
        train_dataloader, valid_dataloader: The splited dataloader of train and valid using for training.
        
        test_dataloader: The dataloader for testing.
    """ 
    logger.info("Building dataloader...")

    logger.info("Loading mean and std value...")
    try:
       
        with open("model/normalize_data.json", "r") as f:
            d = json.load(f)
        mean = d["mean"]
        std = d["std"]
    except:
        if not training:
            raise RuntimeError("No normalize_data.json found. Try training first.")
        logger.warning("Can not found normalize_data.json. Caculating from training data...")
        mean, std = get_mean_and_std(data_path, logger)
        d = {"mean": mean, "std": std}
        os.makedirs("model", exist_ok=True)
        with open("model/normalize_data.json", "w") as f:
            json.dump(d, f)
    logger.info("Normalize data loaded.")

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
        
    dataset = RafDbDataset(data_path, logger=logger, training=training, transform=transform)
    g = torch.Generator().manual_seed(42)
    
    logger.info("Crating datalodaer...")
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
