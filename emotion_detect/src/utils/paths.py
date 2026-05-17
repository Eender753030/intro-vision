from pathlib import Path
import sys


# Get the root directory of the emotion_detect project
# Assume this file is at: emotion_detect/src/utils/paths.py
ROOT = Path(__file__).resolve().parent.parent.parent

# Core Directories
SRC_DIR = ROOT / "src"
MODEL_DIR = ROOT / "model"
LOG_DIR = ROOT / "log"
CONFIG_PATH = ROOT / "config.toml"

# Ensure essential directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def add_src_to_path():
    """Adds the src directory to sys.path if not already present."""
    src_str = str(SRC_DIR)
    if src_str not in sys.path:
        sys.path.append(src_str)


def get_model_path(filename: str) -> str:
    """Returns absolute path for a model file."""
    return str(MODEL_DIR / filename)


def get_log_path(filename: str) -> str:
    """Returns absolute path for a log file."""
    return str(LOG_DIR / filename)
