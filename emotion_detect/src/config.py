import toml
from utils.paths import CONFIG_PATH


def get_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            config = toml.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: {CONFIG_PATH} not found.")
        return {}
            
            
if __name__ == "__main__":
    config = get_config()
    
    print(f"{config}")