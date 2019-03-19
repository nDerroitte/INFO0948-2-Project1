import numpy as np

def mutual_information(joint_prob, p1, p2):
    p12 = np.multiply(p1[:, None], p2[None, :])
    p_joint_divided = joint_prob / p12[:,None]
    log2_p = (np.ma.log2(p_joint_divided)).filled(0)
    prod_entropy = np.multiply(joint_prob, log2_p)
    H = np.sum(prod_entropy)
    return H
