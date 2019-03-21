import numpy as np

def cond_proba(joint_prob, p2):
    """
    Compute P(X|Y) from P(X,Y) and P(Y)
    """
    return joint_prob / p2[:, None]

def cond_joint_proba(joint_prob, p3):
    """
    Compute P(X,Y|Z) from P(X,Y,Z) and P(Z).
    /!\ Only works with 2 variables cond with a third.
    """
    out = np.array([joint_prob[0]/p3[0], joint_prob[1]/p3[1]])
    return out
