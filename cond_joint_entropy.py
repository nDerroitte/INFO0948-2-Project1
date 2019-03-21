import numpy as np

def cond_joint_entropy(joint_prob, pcond):
    """
    Compute H(X,Y|Z), the conditional joint entropy given the joint probability
    P(X,Y,Z) and the conditional probability P(X,Y|Z)
    """
    # Computing log2(P(X,Y|Z))
    log2_p = (np.ma.log2(pcond)).filled(0)
    # Multipling element wise the arrays
    prod_entropy = np.multiply(joint_prob, log2_p)
    # Getting the - sum of the resulting array.
    H = -(np.sum(prod_entropy))
    return H
