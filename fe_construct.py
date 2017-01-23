##### fe_construct.py #####

import numpy as np

def kron_del(i,j):
    if i == j:
        return 1.0
    else:
        return 0.0

def exp_term(r_vec_i, r_vec_j, cn_vec_i, cn_vec_j, theta_r, theta_cn):
    return (np.exp(-0.5*((((r_vec_i - r_vec_j)**2)/(theta_r**2)) + (((cn_vec_i - cn_vec_j)**2)/(theta_cn**2)))))


def ds(k1, k2, r_vec_i, r_vec_j, cn_vec_i, cn_vec_j):
    if k1 == 0:
        ds_k1 = r_vec_i - r_vec_j
    elif k1 == 1:
        ds_k1 = cn_vec_i - cn_vec_j
    if k2 == 0:
        ds_k2 = r_vec_i - r_vec_j
    elif k2 == 1:
        ds_k2 = cn_vec_i - cn_vec_j

    return (ds_k1*ds_k2)


def k_star(r_vec, cn_vec, theta_r, theta_cn, chi, xi_star_vec, num_cv=2):
    k_star_vec = []

    for i in range(r_vec.shape):
        dk_dr_i = (-1.0)*(chi**2)*(r_vec[i] - xi_star_vec[0])*exp_term(r_vec[i], xi_star_vec[0], cn_vec[i], xi_star_vec[1], theta_r, theta_cn)/(theta_r**2)
        dk_dcn_i = (-1.0)*(chi**2)*(cn_vec[i] - xi_star_vec[1])*exp_term(r_vec[i], xi_star_vec[0], cn_vec[i], xi_star_vec[1], theta_r, theta_cn)/(theta_cn**2)
        k_star_vec.append(dk_dr_i)
        k_star_vec.append(dk_dcn_i)

    return np.array(k_star_vec)


# Populating K matrix
def K_construct(r_vec, cn_vec, theta_r, theta_cn, chi, num_cv=2):
    # The resulting matrix' size is nD x nD from n^2 D x D submatrices
    dd_submatrices = []
    for i in range(r_vec.shape):
        for j in range(cn_vec.shape):
            # In this loop, create DxD matrices and append to a list
            dd_submatrix = []
            for ii in range(num_cv):
                for jj in range(num_cv):
                    fac = (chi**2)/((theta_r**2) * (theta_cn**2))
                    if ii % 2 == 0:
                        delta = (kron_del(ii, jj)*(theta_r**2))-ds(ii, jj, r_vec[i], r_vec[j], cn_vec[i], cn_vec[j])
                    else:
                        delta = (kron_del(ii, jj)*(theta_cn**2))-ds(ii, jj, r_vec[i], r_vec[j], cn_vec[i], cn_vec[j])
                    expo = exp_term(r_vec[i], r_vec[j], cn_vec[i], cn_vec[j], theta_r, theta_cn)

                    dd_submatrix.append(fac*delta*expo)
             
            # Reshape dd_submatrix into DxD matrix
            dd_submatrix = np.reshape(np.array(dd_submatrix), (num_cv, num_cv))
            dd_submatrices.append(dd_submatrix)

    # Reshape dd_submatrices into nD x nD matrix
    return np.reshape(np.array(dd_submatrices), (cn_vec.shape*num_cv, r_vec.shape*num_cv))


def fe_fit(r_vec, cn_vec, theta_r, theta_cn, chi, icf_vec, icf_var_vec, xi_star_vec):
    # ICF's size: nD x 1
    # icf_var_vec's size: nD x 1

    var_mat = np.diag(icf_var_vec)
    K = K_construct(r_vec, cn_vec, theta_r, theta_cn, chi)

    inv_term = np.inv(K + var_mat)
    icf_trans_vec = np.matmul(inv_term, icf_vec)

    k_star_vec = k_star(r_vec, cn_vec, theta_r, theta_cn, chi, xi_star_vec)

    free_energy = np.dot(k_star_vec, icf_trans_vec)

    # Returning the fitted free energy and the actual C.N. coordinates as tuple
    return (free_energy, xi_star_vec)
