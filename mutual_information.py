import numpy as np

def mutual_information(joint_prob, p1, p2):
    p12 = np.zeros((p2.size, p1.size))
    for i in range(p1.size):
        for j in range(p2.size):
            p12[j][i] = p1[i] * p2[j]
    p_joint_divided = joint_prob / p12
    log2_p = (np.ma.log2(p_joint_divided)).filled(0)
    prod_entropy = np.multiply(joint_prob, log2_p)
    H = np.sum(prod_entropy)
    return H
