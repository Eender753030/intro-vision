import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor):
        residual = x
        
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        
        x += residual
        return self.relu(x)
    
class ConvBlock(nn.Module):
    def __init__(
        self, 
        n_channels: int, 
        n_filters: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_kernel_size: int = 2,    
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size),
        )
        
    def forward(self, x: torch.Tensor):
        return self.conv_block(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBlock(1, 16),
            ResBlock(16),
            ResBlock(16),
        ) 
        
        self.stage2 = nn.Sequential(
            ConvBlock(16, 32),
            ResBlock(32),
            ResBlock(32),
        ) 
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.out = nn.Linear(32, num_classes)
        
    def forward(self, x: torch.Tensor):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.out(x)
    
    
    
if __name__ == "__main__":
    from torchinfo import summary
    model = SimpleCNN()
    
    summary(model, input_size=(1, 1, 48, 48))
    

        
        
    