import numpy as np

def mutual_information(joint_prob, p1, p2):
    """
    Computes I(X ; Y), the mutual information between X and Y given the joint
    probability of the two variables and the probability distribution of each
    variables.
    """
    # Creating the array of p1 * p2
    p12 = np.zeros((p2.size, p1.size))
    for i in range(p1.size):
        for j in range(p2.size):
            p12[j][i] = p1[i] * p2[j]

    # Computing the argument of the log, ie (joint_prob / (Px * Py))
    p_joint_divided = joint_prob / p12
    # Taking the log of the term
    log2_p = (np.ma.log2(p_joint_divided)).filled(0)
    # Multipling element wise the arrays
    prod_entropy = np.multiply(joint_prob, log2_p)
    # Getting the sum of the resulting array.
    H = np.sum(prod_entropy)
    return H
