##### icf.py #####

# CONSTRUCT ICF VECTOR ON DATAPOINT BASIS
# Algorithm implemented according to Gabor's paper JCTC 2016
# and free energy book

# f = -(inv(G_W)W)(grad(V)) + (\beta)^(-1)*div_r(inv(G_W)W)
# Dimension of f is D x 1

# The most complicated part is calculation of divergence!

# Common matrix sizes
# W = (\grad CVs)^T (\inv(\mu)), \mu is a diagonal mass matrix, so inverse is 
# easy to calculate! (\grad CVs)^T = (D x 3N)
# W's size is (D x 3N)

# G_w = W(\grad CVs) = (D x 3N) x (3N x D) = (D x D)
# grad(V) is (3N x 1)

# grad(CVs) is basically [grad(r_ab) grad(CN)]^T, since we defined each vector to be column vector!
# Make sure dimensions are correct!

import numpy as np

from .colvars.params import RParams, CNParams
from .colvars.distance import grad_x as rgrad_x
from .colvars.distance import hess_x_j as rhess_x_j
from .colvars.coordnum import grad_x as cngrad_x
from .colvars.coordnum import hess_x_j as cnhess_x_j


def inv_mu(mu):
    return np.linalg.inv(mu)


def w_construct(mu, grad_cv_t):
    mu_inv = inv_mu(mu)                         # 3N x 3N
    return np.matmul(grad_cv_t, mu_inv)         # D x 3N


def g_w_construct(W, grad_cv):
    return np.matmul(W, grad_cv)                # D x D


def icf_construct(X_m, mu, grad_V, kT, r_params, cn_params, L):
    # Use inner methods 'firstterm' and 'secondterm'
    # grad_V is 3N x 1
    # hess_r, hess_cn is 3N x 3N each

    r_a = r_params.a_ind
    r_b = r_params.b_ind

    grad_r = rgrad_x(X_m, r_a, r_b, L)
    grad_cn = cngrad_x(X_m, cn_params, L)

    grad_cv_t = np.array([grad_r, grad_cn])     # D x 3N

    W = w_construct(mu, grad_cv_t)
    G_w = g_w_construct(W, grad_cv_t.T)
    G_w_inv = np.linalg.inv(G_w)

    def firstterm(W, G_w_inv, grad_V):
        return (-1.0)*(np.matmul(np.matmul(G_w_inv,W),grad_V))   # D x 1

    def secondterm(X_m, r_a, r_b, cn_a, cn_b_list, L, G_w_inv, W, mu, grad_cv, num_cv=2):
        sum_divergence = np.zeros(num_cv)        # D x 1
        for i in range(X_m.shape):
            hess_x_i_r = rhess_x_j(X_m, r_a, r_b, i, L)
            hess_x_i_cn = cnhess_x_j(X_m, i, cn_params, L)

            hess_cv_t = np.array([hess_x_i_r, hess_x_i_cn])

            first_subterm = np.matmul(np.matmul(G_w_inv, hess_cv_t), inv_mu(mu))    # D x 3N
            
            # Second subterm parts
            second_subterm_innermost = np.matmul(W, hess_cv_t.T) + np.matmul(np.matmul(hess_cv_t, inv_mu(mu)), grad_cv)
            second_subterm_front = np.matmul(np.matmul(G_w_inv*(-1.0), second_subterm_innermost), G_w_inv)
            second_subterm = np.matmul(second_subterm_front, W)     # D x 3N

            d_GwinvW_dxi = first_subterm + second_subterm
            sum_divergence += d_GwinvW_dxi[:, i]

        return (kT * sum_divergence)        # D x 1

    return firstterm(mu, grad_r, grad_cn, grad_V) + secondterm(X_m, r_a, r_b, cn_a, cn_b_list, L, G_w_inv, W, mu, grad_cv_t.T, num_cv=2)    # D x 1
