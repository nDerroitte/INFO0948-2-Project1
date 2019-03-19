import numpy as np

def cond_proba(joint_prob, p2):
    return joint_prob / p2[:, None]

def cond_joint_proba(joint_prob, p3):
    out = np.array([joint_prob[0]/p3[0], joint_prob[1]/p3[1]])
    return out
