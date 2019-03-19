import numpy as np

def cond_joint_entropy(joint_prob, pcond):
    log2_p = (np.ma.log2(pcond)).filled(0)
    prod_entropy = np.multiply(joint_prob, log2_p)
    H = -(np.sum(prod_entropy))
    return H
