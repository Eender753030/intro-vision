import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution implemented.

    Args:
        in_channels: Amount of input channel.
        out_channels: Amount of output channel.
        kernel_size: Size of filter/kernel.
        padding: Amount of padding that around input tensor.
    """
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        final_act: bool = True
    ):
        super().__init__()
        
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                in_channels, 
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if final_act:
            self.pw_conv.append(nn.ReLU6(inplace=True))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x
        
        
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
        use_dwconv: bool = False  
    ):
        super().__init__()
          
        self.conv_block = nn.Sequential(
            nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU6(inplace=True),
            *(
                (DepthwiseSeparableConv(n_filters, n_filters, kernel_size=kernel_size, padding=padding),)
                if use_dwconv else 
                (
                    nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, bias=False),
                    nn.BatchNorm2d(n_filters),
                    nn.ReLU6(inplace=True),
                )
            ),
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
    def __init__(
        self, 
        channels: int, 
        kernel_size: int = 3, 
        padding: int = 1,
        use_dwconv: bool = False,
    ):
        super().__init__()
                
        self.conv1 = (
            DepthwiseSeparableConv(channels, channels, kernel_size=kernel_size, padding=padding) 
            if use_dwconv else 
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True)
            ) 
        )
        self.conv2 = (
            DepthwiseSeparableConv(channels, channels, kernel_size=kernel_size, padding=padding, final_act=False) 
            if use_dwconv else 
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(channels),
            )
        )
        
        self.act = nn.ReLU6(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return self.act(x)


class SimpleCNN(nn.Module):
    """
    Simple CNN model.
    
    Args:
        num_classes: Amount of output classes.
        num_stages: Amount of stage containing ConvBlock + ResBlock.
        num_res_block: Amount of ResBlock in each stage.
        dropout: Dropout rate.
    """
    def __init__(
        self, 
        num_classes: int = 7, 
        num_stages: int = 3, 
        num_res_blocks: int = 2, 
        dropout: float = 0.5,
        base_channels: int = 32,
        use_dwconv: bool = False
    ):
        super().__init__()
        
        # Initial stem
        self.stem = ConvBlock(1, base_channels, pool_kernel_size=2, use_dwconv=use_dwconv)
        
        # Main stages
        stages_list = []
        in_ch = base_channels
        for _ in range(num_stages - 1):
            out_ch = in_ch * 2
            stages_list.append(ConvBlock(in_ch, out_ch, pool_kernel_size=2, use_dwconv=use_dwconv))
            for _ in range(num_res_blocks):
                stages_list.append(ResBlock(out_ch, use_dwconv=use_dwconv))
            in_ch = out_ch
        
        self.stages = nn.Sequential(*stages_list)
        
        # Global Pooling and Classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(in_ch, num_classes)
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
