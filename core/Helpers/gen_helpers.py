import numpy as np
from matplotlib import pyplot as plt

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

def repeat2d_r(x: np.ndarray, ensemble_size: int, batch_size: int) -> np.ndarray:
    """
    Repeat a 2D array (n, m) to a new array (n * ensemble_size, m)
    :param x: array to repeat
    :param ensemble_size: number of times to tile each element
    :param batch_size: total batch size
    :return: read-only repeated array
    """
    if x.shape[0] > batch_size:
        raise ValueError('Array length cannot be greater than batch size')
    elif batch_size % ensemble_size != 0:
        raise ValueError('Batch size must be divisible by ensemble size')
    expanded_x = x[:, np.newaxis, :]
    tiled_x_r = np.broadcast_to(expanded_x, (x.shape[0], ensemble_size, x.shape[1]))
    return tiled_x_r.reshape(x.shape[0] * ensemble_size, x.shape[1])

def plot(x: np.ndarray, y: np.ndarray, scatter: bool = False, title: str = None, labels: tuple = None, lims: list = None, hlines: tuple = None, tight: bool = True) -> None:
    if scatter:
        plt.scatter(x, y)
    else:
        plt.plot(x, y)

    if title is not None:
        plt.title(title)

    if labels is not None:
        plt.xlabel(labels[0])
        if len(labels) > 1:
            plt.ylabel(labels[1])

    if lims is not None:
        plt.xlim(*lims[0])
        if len(lims) > 1:
            plt.ylim(*lims[1])

    if hlines is not None:
        plt.hlines(*hlines, linestyle='--', color='r')

    if tight:
        plt.tight_layout()
    plt.show()