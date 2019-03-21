import numpy as np
from util import *
from entropy import *
from joint_entropy import *
from cond_joint_entropy import *
from mutual_information import *
from conditional_entropy import *
from cond_mutual_information import *

# Constants
p_joint_xy = np.array([[1/8, 1/16, 1/16, 1/4], [1/16, 1/8, 1/16, 0],
                      [1/32, 1/32, 1/16, 0], [1/32, 1/32, 1/16, 0]])
p_xy_w = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1,0], [0, 0, 0, 1]])
p_xy_z = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0,1], [1, 1, 1, 0]])


if __name__ == "__main__":
    print("Precomputation ...", end="")
    size_array = p_joint_xy.size
    # Computing the probability distribution :
    p_xi = p_joint_xy.sum(axis=0)
    p_yi = p_joint_xy.sum(axis=1)
    p_wi = np.zeros(2)
    p_zi = np.zeros(2)
    for i in range(4):
        for j in range(4):
            if(p_xy_w[i][j] == 0):
                p_wi[1] += p_joint_xy[i][j]
                p_zi[0] += p_joint_xy[i][j]
            else:
                p_wi[0] += p_joint_xy[i][j]
                p_zi[1] += p_joint_xy[i][j]

    # Computing the joint distribution of the variable we don't have
    # Initialization
    p_joint_xw = np.zeros((2, 4))
    p_joint_xz = np.zeros((2, 4))
    p_joint_yw = np.zeros((2, 4))
    p_joint_yz = np.zeros((2, 4))
    # Computation of the values
    p_joint_xw[0] = (p_joint_xy * p_xy_w).sum(axis=0)
    p_joint_xw[1] = (p_joint_xy * p_xy_z).sum(axis=0)
    p_joint_xz[0] = (p_joint_xy * p_xy_w).sum(axis=0)
    p_joint_xz[1] = (p_joint_xy * p_xy_z).sum(axis=0)
    p_joint_yw[0] = (p_joint_xy * p_xy_z).sum(axis=1)
    p_joint_yw[1] = (p_joint_xy * p_xy_w).sum(axis=1)
    p_joint_yz[0] = (p_joint_xy * p_xy_z).sum(axis=1)
    p_joint_yz[1] = (p_joint_xy * p_xy_w).sum(axis=1)
    p_joint_wz = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            if i == j:
                p_joint_wz[i][j] = 0
            else:
                p_joint_wz[i][j] = p_wi[i]
    # Computing the joint distribution of 3 variables for the 2 cases we need
    # 1 : P(X,Y,W)
    p_joint_xyw = np.zeros((2,4,4))
    for i in range(4):
        for j in range(4):
            if i == j:
                p_joint_xyw[0][i][j] = 0
                p_joint_xyw[1][i][j] = p_joint_xy[i][j]
            else:
                p_joint_xyw[0][i][j] =  p_joint_xy[i][j]
                p_joint_xyw[1][i][j]= 0
    # 2 : P(W,Z,X)
    p_joint_wzx = np.zeros((2,2,4))
    for i in range(2):
        for j in range(4):
            if i ==0 :
                p_joint_wzx[0][i][j] = p_joint_xw[i][j]
                p_joint_wzx[1][i][j] = 0
            else :
                p_joint_wzx[0][i][j] = 0
                p_joint_wzx[1][i][j] = p_joint_xw[i][j]
    print(" done!")

    print("Verification of the Q1: ")

    Hx = entropy(p_xi)
    print("H(X) = {}".format(Hx))

    Hy = entropy(p_yi)
    print("H(Y) = {}".format(Hy))

    Hw = entropy(p_wi)
    print("H(W) = {}".format(Hw))

    Hz = entropy(p_zi)
    print("H(Z) = {}".format(Hz))

    print("Verification of the Q2: ")
    Hxy = joint_entropy(p_joint_xy)
    print("H(X,Y) = {}".format(Hxy))

    Hxw = joint_entropy(p_joint_xw)
    print("H(X,W) = {}".format(Hxw))

    Hyw = joint_entropy(p_joint_yw)
    print("H(Y,W) = {}".format(Hyw))

    Hwz = joint_entropy(p_joint_wz)
    print("H(W,Z) = {}".format(Hwz))

    print("Verification of the Q3: ")

    Hxcondy = cond_entropy(p_joint_xy, cond_proba(p_joint_xy, p_yi))
    print("H(X|Y) = {}".format(Hxcondy))


    Hwcondx = cond_entropy(p_joint_xw.T, cond_proba(p_joint_xw.T, p_xi))
    print("H(W|X) = {}".format(Hwcondx))

    Hzcondx = cond_entropy(p_joint_xz.T, cond_proba(p_joint_xz.T, p_xi))
    print("H(Z|X) = {}".format(Hzcondx))

    Hzcondw = cond_entropy(p_joint_wz, cond_proba(p_joint_wz, p_wi))
    print("H(Z|W) = {}".format(Hzcondw))

    Hwcondz = cond_entropy(p_joint_wz.T, cond_proba(p_joint_wz.T, p_zi))
    print("H(W|Z) = {}".format(Hwcondz))

    print("Verification of the Q4: ")

    Hxycondw = cond_joint_entropy(p_joint_xyw, cond_joint_proba(p_joint_xyw,
                                                                p_wi[::-1]))
    print("H(X,Y|W) = {}".format(Hxycondw))

    Hwzcondx = cond_joint_entropy(p_joint_wzx, cond_joint_proba(p_joint_wzx,
                                                                p_xi))
    print("H(W,Z|X) = {}".format(Hwzcondx))

    print("Verification of the Q5: ")
    # The verifiction using the simple equation is not implemented since it's
    # really straigth forward and can be check directly.
    Ixy = mutual_information(p_joint_xy, p_xi, p_yi)
    print("I(X;Y) = {}".format(Ixy))

    Ixw = mutual_information(p_joint_xw, p_xi, p_wi)
    print("I(X;W) = {}".format(Ixw))

    Iyz = mutual_information(p_joint_yz, p_yi, p_zi)
    print("I(Y;Z) = {}".format(Iyz))

    Iwz = mutual_information(p_joint_wz.T, p_wi, p_zi)
    print("I(W;Z) = {}".format(Iwz))

    print("Verification of the Q6: ")

    # For the Q6, we verify each of the equation given in the report.
    # We test the first one in the first case and the second one in the
    # second case. This can be changed obviously and use the same equation for
    # the two results since they are equivalent.
    Hxyw = joint_entropy(p_joint_xyw)
    Ixycondw =  Hxw + Hyw - Hxyw - Hz
    print("I(X;Y|W) = {}".format(Ixycondw))

    Iwzcondx = cond_mutual_information(p_joint_wzx,
                                   cond_joint_proba(p_joint_wzx,
                                                    p_xi),
                                                    cond_proba(p_joint_xw.T,
                                                               p_xi).T,
                                                    cond_proba(p_joint_xz.T,
                                                               p_xi).T)
    print("I(W;Z|X) = {}".format(Iwzcondx))
