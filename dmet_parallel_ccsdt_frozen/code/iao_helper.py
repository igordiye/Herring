#!/usr/bin/python

import numpy as np
import scipy.linalg as sla


def symm_orthog (X, S):
    Sx = np.dot(X.T, np.dot(S, X))
    val, vec = sla.eigh(-Sx)
    idx = -val > 1.e-12
    Ut = np.dot(vec[:,idx]*1./(np.sqrt(-val[idx])), vec[:,idx].T)
    return np.dot(X, Ut)

#   ----- iao = iao_helper.build_iao (Sf, cf_loc, cf0) -----
def build_iao (S, C_oc, P_valence, P_core=None, P_virt=None):
    # C_oc      :: occupied orbitals
    # P_valence :: valence basis functions
    # S         :: overlap matrix

    assert (P_core is None and P_virt is None)

    nb = S.shape[0]

    # find IAO's
    # [ B2 is a subset of B1 ]
    # symmetric orthonormalization of B2 subset
    # P_valence corresponds to the small basis B2 of Gerald's 2013 paper
    Px = symm_orthog (P_valence, S)

    # project occupied orbitals into B2 subset
    # find orthonormal set C_oc_p
    M = np.dot(np.dot(Px.T, S), C_oc)
    Q, R = sla.qr (M, mode='economic')
    #this operation is like doing _P12 P_21 |i) in Gerald's paper
    C_oc_p = np.dot(Px, Q)

    # apply iao construction
    #M1 is Gerald's operator O (projector onto occupied HF states)
    M1 = np.dot(np.dot(C_oc, C_oc.T), S)
    #M2 is Gerald's operator O tilde (projector onto occupied HF states projected on the basis B2)
    M2 = np.dot(np.dot(C_oc_p, C_oc_p.T), S)
    #this is Eq 2 of Gerald's paper
    At = np.dot(np.dot(M1, M2), Px) \
       + np.dot(np.dot(np.eye(nb)-M1, np.eye(nb)-M2), Px)
    A  = symm_orthog (At, S)
    A_valence = A

    return A_valence

