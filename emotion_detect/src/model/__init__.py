from .model import SimpleCNN
from .mobile_net_v4 import MobileNetV4

def get_model(config: dict) -> SimpleCNN:
    model_config = config["model"]

    return SimpleCNN (
        num_classes=model_config["num_classes"],
        num_stages=model_config["num_stages"],
        num_res_blocks=model_config["num_res_blocks"],
        dropout=model_config["dropout"],
        base_channels=model_config["base_channels"],
        use_dwconv=model_config["use_dwconv"]
    )
