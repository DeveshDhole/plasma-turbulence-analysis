"""Small helpers used by several notebooks."""
import re

import numpy as np


def index_of_just_smaller(arr, value):
    arr = np.asarray(arr)
    mask = arr < value
    if not np.any(mask):
        return None  # No smaller value exists
    return np.argmax(np.where(mask, arr, -np.inf))


def extract_timestep(fname):
    return int(re.search(r'_t(\d+)\.h5$', fname).group(1))
