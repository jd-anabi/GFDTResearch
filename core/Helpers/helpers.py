import os
import sys

import numpy as np
import torch

def rescale(x: torch.Tensor | float, x_scale: torch.Tensor | float, x_offset: torch.Tensor | float = 0) -> torch.Tensor:
    """
    Rescale a tensor: x_dim = x_scale * x + x_offset.
    Supports batched rescaling via broadcasting (e.g. x_scale shape (batch, 1) with x shape (batch, T) or (1, T)).

    :param x: tensor or float to rescale
    :param x_scale: scale factor (scalar or tensor)
    :param x_offset: offset (scalar or tensor); default is 0
    :return: rescaled tensor
    """
    return x_scale * x + x_offset

def get_even_ids(l: int, n: int) -> list:
    """
    Get evenly spaced indices from an array
    :param l: length of array
    :param n: number of evenly spaced indices
    :return: list of evenly spaced indices
    """
    # edge cases
    if n > l:
        raise ValueError('Number of evenly spaced indices cannot be greater than length of array')
    elif n <= 0:
        return []
    elif n == 1:
        return [0]
    ids = [round(i * (l - 1) / (n - 1)) for i in range(n)]
    ids[-1] = l
    return ids

def concat(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Concatenate two arrays with the same number of rows
    :param x: first array
    :param y: second array
    :return: concatenated array
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError('Both arrays must have same number of rows')
    return np.concatenate((x, y), axis=1)

def clear_screen() -> None:
    """
    Clears the console screen depending on the operating system.

    This function checks the underlying operating system and executes the appropriate
    command to clear the terminal screen. It supports both Windows and Unix-based
    platforms.

    :return: None
    """
    if sys.platform == 'win32':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

