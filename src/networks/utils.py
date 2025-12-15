"""Utility functions for neural networks."""

import torch
import torch.nn as nn
from typing import Optional


def init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """
    Initialize weights using orthogonal initialization.
    
    Args:
        module: PyTorch module to initialize
        gain: Scaling factor for orthogonal initialization
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def get_device(device: Optional[str] = "auto") -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device: Device specification ("auto", "cuda", "cpu")
        
    Returns:
        torch.device object
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    else:
        return torch.device("cpu")
