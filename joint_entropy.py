import numpy as np

def joint_entropy(joint_prob):
    """
    Computes H(X , Y), the joint entropy of X and Y given the joint probability
    of X and Y.
    """
    # Computing log2(P joint)
    log2_p = (np.ma.log2(joint_prob)).filled(0)
    # Multipling element wise the arrays
    prod_entropy = np.multiply(joint_prob, log2_p)
    # Getting the - sum of the resulting array.
    H = -( np.sum(prod_entropy))
    return H
