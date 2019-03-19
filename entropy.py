import numpy as np

def entropy(pi):
    log2_pi = np.log2 (pi)
    prod_entropy = np.multiply(pi, log2_pi)
    H = - (np.sum(prod_entropy))
    return H
