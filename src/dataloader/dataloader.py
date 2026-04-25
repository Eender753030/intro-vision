import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import io
from torchvision.transforms import v2

class RafDbDataset(Dataset):
    def __init__(self, data_path: os.PathLike, training: bool = True, transform: v2.Transform = None):
        super().__init__()
        csv_root = "train_labels.csv" if training else "test_labels.csv"
        datalist = pd.read_csv(os.path.join(data_path, csv_root))
        self.img_name = list(datalist['image'])
        self.label = list(datalist['label'])
        self.transform = transform 
        self.dataroot = os.path.join(data_path, "DATASET")
        if training:
            self.dataroot = os.path.join(self.dataroot, "train")
        else:
            self.dataroot = os.path.join(self.dataroot, "test")
       
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
     