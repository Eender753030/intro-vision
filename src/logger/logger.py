import logging
import os
import sys
from termcolor import colored

def get_logger(name: str, target_dir: os.PathLike = "."):
    """
    Create a logger with StreamHandler for stdout and FileHandler making `name`.log to target directory.

    Args:
        name (str): Logger's name.
        target_dir (os.PathLike, optional): Destination directory for log file. Defaults to CWD.

    Returns:
        logging.Logger: Instance of logger
    """
    os.makedirs(target_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    datefmt = "%Y-%m-%d %H:%M:%S"
    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d) %(levelname)s: %(message)s"
    color_fmt = colored("[%(asctime)s %(name)s] ", "green") \
        + colored("(%(filename)s %(lineno)d) ", "yellow") \
        + "%(levelname)s: %(message)s"
        
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(color_fmt, datefmt=datefmt))
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(os.path.join(target_dir, name + ".log"), mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(file_handler)
    
    logger.info("Logger build completed")
    
    return logger