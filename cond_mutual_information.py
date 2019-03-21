import numpy as np

def cond_mutual_information(joint_prob, pcond1, pcond2, pcond3):
    """
    Compute I(X;Y|Z), the conditional joint entropy given the joint probability
    P(X,Y,Z), the conditional probability P(X,Y|Z) and the two conditional
    probability : P(X|Z) and P(Y|Z).
    """

    # Comptuing P(X|Z)*P(Y|Z)
    p23 = pcond2 * pcond3
    # Computing the argument of the log2
    p_joint_divided = (np.ma.divide(pcond1, p23[:, None])).filled(0)
    # Comuting the log2 of the term
    log2_p = (np.ma.log2(p_joint_divided)).filled(0)
    # Multipling the arrays together element wise
    prod_entropy = np.multiply(joint_prob, log2_p)
    # Summing the resulting array
    H = (np.sum(prod_entropy))
    return H
