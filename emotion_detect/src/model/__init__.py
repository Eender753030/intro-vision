from .model import SimpleCNN


def get_model(config: dict) -> SimpleCNN:
    model_config = config["model"]
    return SimpleCNN (
        num_class=model_config["num_class"],
        num_stage=model_config["num_stage"],
        num_res=model_config["num_res"],
        dropout=model_config["dropout"],
        base_channels=model_config.get("base_channels", 32)
    )
