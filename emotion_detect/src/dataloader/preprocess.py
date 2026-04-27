import os
import torch
import numpy as np
import pandas as pd
from logging import Logger
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from .dataset import RafDbDataset


def get_classes_weight(data_path: os.PathLike) -> torch.Tensor:
    """
    Caculate the weight of every class in the dataset.
    Less amount has greater weight.
    """
    df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    total = len(df["label"])
    weights = df["label"].value_counts().sort_index().apply(lambda x: 1 / (x/total)).tolist()

    return torch.tensor(weights)


def get_mean_and_std(data_path: os.PathLike, logger: Logger) -> tuple[float, float]:
    """
    Load all training data and caculate mean and std values.
    """
    tmp_transform = v2.Compose([
        v2.Grayscale(),
        v2.Resize((48, 48)),
        v2.ToImage(),                           
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    tmp_dataset = RafDbDataset(data_path, logger=logger, training=True, transform=tmp_transform)
    
    loader = DataLoader(tmp_dataset, batch_size=256)
    
    total_pixel = 0
    mean = 0.0
    sq_mean = 0.0
    for batch, _ in loader:
        B, C, H, W = batch.shape
        
        total_pixel += B * C * H * W
        mean += batch.sum()
        sq_mean += (batch**2).sum()
        
    mean /= total_pixel
    sq_mean /= total_pixel
    std = torch.sqrt(sq_mean - mean**2)

    return mean.item(), std.item()
