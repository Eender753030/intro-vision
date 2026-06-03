import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic convolution block.

    Args:
        n_channels: Amount of input channel.
        n_filters: Amount of output channel.
        kernel_size: Size of filter/kernel.
        padding: Amount of padding that around input tensor.
        pool_kernel_size: Size of pool's kernel.
    """
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
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class ResBlock(nn.Module):
    """
    Two convolution unit with input residual pass to output.
    
    Args:
        channels: Amount of input and ouput channel.
        kernel_size: Size of filter/kernel.
        padding: Amount of padding that around input tensor.
    """
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
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act1(self.conv1(x))
        x = self.conv2(x)
        x += residual
        return self.act2(x)


class SimpleCNN(nn.Module):
    """
    Simple CNN model.
    
    Args:
        num_class: Amount of output classes.
        num_stage: Amount of stage containing ConvBlock + ResBlock.
        num_res: Amount of ResBlock in each stage.
        dropout: Dropout rate.
    """
    def __init__(
        self, 
        num_class: int = 7, 
        num_stage: int = 3, 
        num_res: int = 2, 
        dropout: float = 0.5,
        base_channels: int = 32,
    ):
        super().__init__()
        
        # Initial stem
        self.stem = ConvBlock(1, base_channels, pool_kernel_size=2)
        
        # Main stages
        stages_list = []
        in_ch = base_channels
        for i in range(num_stage - 1):
            out_ch = in_ch * 2
            stages_list.append(ConvBlock(in_ch, out_ch, pool_kernel_size=2))
            for _ in range(num_res):
                stages_list.append(ResBlock(out_ch))
            in_ch = out_ch
        
        self.stages = nn.Sequential(*stages_list)
        
        # Global Pooling and Classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_class)
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stages(x)
        
        # Features after global pooling
        features = self.pool(x)
        features = features.view(features.size(0), -1)
        
        # Logits for classification
        logits = self.fc(features)
        
        return logits, features
    
    
if __name__ == "__main__":
    from torchinfo import summary
    model = SimpleCNN()
    
    summary(model, input_size=(1, 1, 48, 48))
