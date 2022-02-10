import numpy as np


def pseudo_divide(a, b, rank):
    n = b.shape[0]
    b = np.concatenate([np.zeros_like(b[rank:]), b[n-rank:]])
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
