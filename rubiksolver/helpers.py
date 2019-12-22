import numpy as np


def min_max_scaling(arr: np.ndarray, min_val: float=-2.0, max_val: float=2.0):
    """
    scales the given array between 0 and 1
    :param arr: numpy array
    :param min_val: min occurence in data, default is -2.0 based on the min possible stone value
    :param max_val: max occurence in data, default is +2.0 based on the max possible stone value
    :return: scaled numpy array
    """
    return (arr.astype('float32') - min_val) / (max_val - min_val)