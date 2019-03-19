import numpy as np

def joint_entropy(joint_prob):
    log2_p = (np.ma.log2(joint_prob)).filled(0)
    prod_entropy = np.multiply(joint_prob, log2_p)
    H = -( np.sum(prod_entropy))
    return H
