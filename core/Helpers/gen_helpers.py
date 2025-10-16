import numpy as np


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
    return [round(i * (l - 1) / (n - 1)) for i in range(n)]

def concat(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Concatenate two arrays with same number of rows
    :param x: first array
    :param y: second array
    :return: concatenated array
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError('Both arrays must have same number of rows')
    return np.concatenate((x, y), axis=1)

def sde_tile(x: np.ndarray, ensemble_size: int, batch_size: int) -> np.ndarray:
    """
    Tile an array appropriately for internal SDE solver
    :param x: array to tile
    :param ensemble_size: number of times to tile each element
    :param batch_size: total batch size
    :return: tiled array (size: batch_size)
    """
    if x.shape[0] > batch_size:
        raise ValueError('Array length cannot be greater than batch size')
    tiled_x = np.zeros(batch_size, dtype=x.dtype)
    for i in range(len(x)):
        tiled_x[i * ensemble_size:(i + 1) * ensemble_size] = np.tile(x[i], ensemble_size)
    return tiled_x