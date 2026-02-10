"""
Device detection utility for PyTorch.
"""

import torch


def get_device() -> torch.device:
    """Return the best available PyTorch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
