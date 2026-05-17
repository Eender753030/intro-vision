import os
import torch
import json
from logging import Logger
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

from .dataset import RafDbDataset
from .preprocess import get_mean_and_std
from utils.paths import MODEL_DIR


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
    norm_json_path = MODEL_DIR / "normalize_data.json"
    try:
        with open(norm_json_path, "r") as f:
            d = json.load(f)
        mean = d["mean"]
        std = d["std"]
    except:
        if not training:
            raise RuntimeError("No normalize_data.json found. Try training first.")
        logger.warning("Can not found normalize_data.json. Caculating from training data...")
        mean, std = get_mean_and_std(data_path, logger)
        d = {"mean": mean, "std": std}
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(norm_json_path, "w") as f:
            json.dump(d, f)
    logger.info("Normalize data loaded.")

    if training:
        # Define Training Transform (Augmented)
        train_transform = v2.Compose([
            v2.Grayscale(),
            v2.RandomResizedCrop(size=(48, 48), scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[mean], std=[std]),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        ])

        # Define Validation Transform (Clean)
        val_transform = v2.Compose([
            v2.Grayscale(),
            v2.Resize((48, 48)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[mean], std=[std]),
        ])

        # Create two separate dataset objects pointing to the same data but with different transforms
        full_dataset_train = RafDbDataset(data_path, logger=logger, training=True, transform=train_transform)
        full_dataset_val = RafDbDataset(data_path, logger=logger, training=True, transform=val_transform)

        len_dataset = len(full_dataset_train)
        indices = list(range(len_dataset))
        train_size = int(0.70 * len_dataset)
        
        import random
        random.seed(42)
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        from torch.utils.data import Subset
        train_set = Subset(full_dataset_train, train_indices)
        val_set = Subset(full_dataset_val, val_indices)

        logger.info("Creating dataloaders...")
        train_loader = DataLoader(
            train_set, 
            batch_size=config["data"]["batch_size"], 
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )
        valid_loader = DataLoader(
            val_set, 
            batch_size=config["data"]["batch_size"], 
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )
        return train_loader, valid_loader
    
    # Testing mode (remains similar but with centralized logic if needed)
    test_transform = v2.Compose([
        v2.Grayscale(),
        v2.Resize((48, 48)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[mean], std=[std]),
    ])
    dataset = RafDbDataset(data_path, logger=logger, training=False, transform=test_transform)
    return DataLoader(
        dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
