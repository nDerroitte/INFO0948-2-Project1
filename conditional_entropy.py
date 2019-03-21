import numpy as np

def cond_entropy(joint_prob, cond_prob):
    """
    Computes H(X |Y), the conditional entropy of X given Y given the joint
    probability and the conditional probability of the two variables.
    """
    # Computing log2(P cond)
    log2_p = (np.ma.log2(cond_prob)).filled(0)
    # Multipling element wise the arrays
    prod_entropy = np.multiply(joint_prob, log2_p)
    # Getting the - sum of the resulting array.
    H = -( np.sum(prod_entropy))
    return H
