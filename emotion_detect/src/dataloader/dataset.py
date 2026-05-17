import os
import pandas as pd
from logging import Logger
from typing import Optional
from torch.utils.data import Dataset
from torchvision import io
from torchvision.transforms import v2
from pathlib import Path
from utils.paths import ROOT


class RafDbDataset(Dataset):
    """
    Custom dataset that use RAF-DB.
    Load image in __getitem__ in runtime. 
    
    Args:
        data_path: The root path of dataset.
        logger: The Logger to record information or warning.
        training: Training or testing mode.
        transform (Optional): Pre-process transform before output image.
    """
    def __init__(
        self, 
        data_path: os.PathLike, 
        logger: Logger,
        training: bool = True, 
        transform: Optional[v2.Transform] = None,
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        if not self.data_path.is_absolute():
            self.data_path = (ROOT / data_path).resolve()
        
        csv_path = os.path.join(self.data_path, "train_labels.csv" if training else "test_labels.csv")

        try: 
            datalist = pd.read_csv(csv_path)
        except:
            logger.warning(f"Can not found dataset in {self.data_path}. Start downloading RAF-DB from net.")
            self._download_dataset(self.data_path)
            logger.info(f"Download completed in folder: {self.data_path}")
            datalist = pd.read_csv(csv_path)

        self.img_name = list(datalist['image'])
        self.label = list(datalist['label'])
        self.transform = transform 
        self.dataroot = os.path.join(self.data_path, "DATASET")
        if training:
            self.dataroot = os.path.join(self.dataroot, "train")
        else:
            self.dataroot = os.path.join(self.dataroot, "test")
       
    def _download_dataset(self, target_path: os.PathLike):
        """
        Download the RAF-DB data set from Kaggle.
        Automatic download zip from web and extract the files. 
        """
        import requests
        import zipfile

        target_path = Path(target_path)
        os.makedirs(target_path, exist_ok=True)

        zip_path = target_path / "temp.zip"

        with requests.get("https://www.kaggle.com/api/v1/datasets/download/shuvoalok/raf-db-dataset", stream=True) as r:
            r.raise_for_status()

            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk) 

        with zipfile.ZipFile(zip_path, 'r') as zip:
            zip.extractall(target_path)

        os.remove(zip_path)

    def __len__(self):
        return len(self.img_name)
    
    def __getitem__(self, index):
        label = self.label[index] - 1
        img_path = os.path.join(self.dataroot, str(self.label[index]),self.img_name[index])
        data = io.read_image(img_path)        
        
        if self.transform is not None:
            data = self.transform(data)
            
        return data, label


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random 
    
    transform = v2.Compose([
        v2.Grayscale(),
        v2.Resize((48, 48)),
    ])
    dataset = RafDbDataset("archive", training=True, transform=transform)
    num_data = len(dataset)
    print(f"Total training data: {num_data}")
    img_idx = random.randint(0, num_data)
    data, _ = dataset[img_idx]
    plt.imshow(data[0], cmap="gray")
    plt.title("Random image from training dataset")
    plt.axis("off")
    plt.show()
     