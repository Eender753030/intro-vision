import tomllib
import os


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "config.toml" )


def get_config() -> dict:
    """
    Read configuration settings from config.toml
    """
    with open(CONFIG_FILE, "rb") as f:
        config = tomllib.load(f)  
    return config
            
            
if __name__ == "__main__":
    config = get_config()
    
    print(f"{config}")