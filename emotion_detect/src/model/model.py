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
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size),
        )
        
    def forward(self, x: torch.Tensor):
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
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor):
        residual = x
        
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        
        x += residual
        return self.relu(x)


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
        num_stage: int = 2, 
        num_res: int = 2, 
        dropout: float = 0.5,
    ):
        super().__init__()
        
        channels = [1] + [16 * 2**x for x in range(num_stage)]
        
        self.stages = nn.Sequential(
            *(self._make_stage(channels[i], channels[i + 1], num_res) for i in range(num_stage))
        )
      
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.flatten = nn.Flatten(start_dim=1)
        
        self.out = nn.Linear(channels[num_stage], num_class)
        
    @staticmethod
    def _make_stage(input, output, num_res):
        return nn.Sequential(
            ConvBlock(input, output),
            *(ResBlock(output) for _ in range(num_res)),
        )    
    
    def forward(self, x: torch.Tensor):      
        for stage in self.stages:
            x = stage(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.out(x)
    
    
if __name__ == "__main__":
    from torchinfo import summary
    model = SimpleCNN()
    
    summary(model, input_size=(1, 1, 48, 48))
