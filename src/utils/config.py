import os
import yaml
from pathlib import Path

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Parameters:
    - config_path: str or Path, path to config file (absolute or relative)
    
    Returns:
    - dict: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def get_default_config():
    return load_config(os.path.join(os.path.dirname(__file__), '../../configs/default.yaml'))

def get_split_imagenet_config():
    return load_config(os.path.join(os.path.dirname(__file__), '../../configs/split_imagenet.yaml'))