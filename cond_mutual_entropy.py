import numpy as np

def cond_mutual_entropy(joint_prob, pcond1, pcond2, pcond3):
    p23 = pcond2 * pcond3
    p_joint_divided = (np.ma.divide(pcond1, p23[:, None])).filled(0)
    log2_p = (np.ma.log2(p_joint_divided)).filled(0)
    prod_entropy = np.multiply(joint_prob, log2_p)
    H = (np.sum(prod_entropy))
    return H
