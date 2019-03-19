import numpy as np

def cond_proba(joint_prob, p2):
    return joint_prob / p2[:, None]
