"""Utility functions for conditional text generation."""

import random
import numpy as np
import torch
from typing import Any, Dict, Optional, Union
from pathlib import Path
import yaml
from omegaconf import OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    
    if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = OmegaConf.load(config_path)
        
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save configuration.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def format_text_for_condition(condition: str, prompt: str, separator: str = ": ") -> str:
    """Format text with condition and prompt.
    
    Args:
        condition: The conditioning text.
        prompt: The input prompt.
        separator: Separator between condition and prompt.
        
    Returns:
        Formatted conditional text.
    """
    return f"{condition}{separator}{prompt}"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Input text.
        max_length: Maximum length.
        suffix: Suffix to add when truncating.
        
    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
