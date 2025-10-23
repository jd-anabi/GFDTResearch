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
    #repeated_x = np.zeros((batch_size, x.shape[1]), dtype=x.dtype)
    #repeated_x = np.repeat(x, ensemble_size, axis=0)
    return tiled_x_r.reshape(x.shape[0] * ensemble_size, x.shape[1])