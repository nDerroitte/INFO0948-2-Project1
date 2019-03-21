import numpy as np

def entropy(pi):
    """
    Computes H(X), the entropy of a random variable X , given its probability
    distribution pi = (p1, p2, . . . , pn).
    """
    # Computing log2(P_Xi)
    log2_pi = (np.ma.log2 (pi)).filled(0)
    # Multipling element wise the arrays
    prod_entropy = np.multiply(pi, log2_pi)
    # Getting the - sum of the resulting array.
    H = - (np.sum(prod_entropy))
    return H
