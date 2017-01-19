##### coordnum.py #####

import numpy as np
import sys, os

from ..chemmatrixaux import atomindex_to_vecindex as atv

from .distance import dr_dxi, d2r_dxjxi

from ..pbc import pair_dx


# Coordnum data structure
class CNParams():
   def __init__(self, numer_pow=6.0, denom_pow=12.0, r0=2.0):
      self.numer_pow = numer_pow
      self.denom_pow = denom_pow
      self.r0 = r0


# Auxillary methods for D_ab_k and N_ab_k
def n_ab_k(r_ab_k, r0, numer_pow):
   return (1.0-(r_ab_k/r0)**(numer_pow))


def d_ab_k(r_ab_k, r0, denom_pow):
   return (1.0-(r_ab_k/r0)**(denom_pow))


def dn_dd(exp, r_ab, r0):
   return ((-exp)*((r_ab/r0)**(exp-1)))


def grad_x(X_m, atom_a_index, atom_b_indices, cn_params, L):
   # Simplify variable names
   a = atom_a_index
   b_list = atom_b_indices
   n = cn_params.numer_pow
   m = cn_params.denom_pow
   r0 = cn_params.r0

   # n,d method aliases
   N = n_ab_k
   D = d_ab_k

   x_a = np.array([X_m[atv(a,'x')], X_m[atv(a,'y')], X_m[atv(a,'z')]])

   grad_x_list = []

   for i in range(X_m.shape):
      sum_dcn_dxi = 0.0
      for b in b_list:
         x_b = np.array([X_m[atv(b,'x')], X_m[atv(b,'y')], X_m[atv(b,'z')]])
         r_ab = pair_dx(x_a, x_b, L)

         factor = D(r_ab, r0, m)**(-2.0)
         grad_firstterm = (-n)*D(r_ab, r0, m)*((r_ab/r0)**(n-1.0))*dr_dxi(x_a, x_b, a, b, i, L, r_ab)
         grad_secondterm = (-m)*N(r_ab, r0, n)*((r_ab/r0)**(m-1.0))*dr_dxi(x_a, x_b, a, b, i, L, r_ab)

         sum_dcn_dxi += (factor*(grad_firstterm - grad_secondterm))

      grad_x_list.append(sum_dcn_dxi)

   return np.array(grad_x_list)


def hess_x_j(X_m, atom_a_index, atom_b_indices, vec_index_j, cn_params, L):
   a = atom_a_index
   b_list = atom_b_indices
   j = vec_index_j
   n = cn_params.numer_pow
   m = cn_params.denom_pow
   r0 = cn_params.r0

   N = n_ab_k
   D = d_ab_k

   x_a = np.array([X_m[atv(a,'x')], X_m[atv(a,'y')], X_m[atv(a,'z')]])

   hess_x_j_list = []

   for i in range(X_m.shape):
      sum_d2cn_dxjxi = 0.0
      for b in b_list:
         x_b = np.array([X_m[atv(b,'x')], X_m[atv(b,'y')], X_m[atv(b,'z')]])
         r_ab = pair_dx(x_a, x_b, L)

         factor = D(r_ab, r0, m)**(-4.0)
         hess_t1 = D(r_ab, r0, m) * d2r_dxjxi(x_a, x_b, a, b, i, j, L, r_ab) * dn_dd(n, r_ab, r0)
         hess_t2 = (-n) * D(r_ab, r0, m) * (-1.0) * dn_dd(n-1, r_ab, r0) * dr_dxi(x_a, x_b, a, b, i, L, r_ab) * dr_dxi(x_a, x_b, a, b, j, L, r_ab)
         hess_t3 = dn_dd(n, r_ab, r0) * dr_dxi(x_a, x_b, a, b, i, L, r_ab) * dn_dd(m, r_ab, r0) * dr_dxi(x_a, x_b, a, b, j, L, r_ab)
         hess_t4 = N(r_ab, r0, n) * d2r_dxjxi(x_a, x_b, a, b, i, j, L, r_ab) * dn_dd(m, r_ab, r0)
         hess_t5 = (-m) * N(r_ab, r0, n) * (-1.0) * dn_dd(m-1, r_ab, r0) * dr_dxi(x_a, x_b, a, b, i, L, r_ab) * dr_dxi(x_a, x_b, a, b, j, L, r_ab)
         hess_t6 = dn_dd(m, r_ab, r0) * dr_dxi(x_a, x_b, a, b, i, L, r_ab) * dn_dd(n, r_ab, r0) * dr_dxi(x_a, x_b, a, b, j, L, r_ab)

         group1 = hess_t1 + hess_t2 + hess_t3
         group2 = hess_t4 + hess_t5 + hess_t6

         sum_d2cn_dxjxi += (factor*(group1 - group2))

      hess_x_j_list.append(sum_d2cn_dxjxi)

   return np.array(hess_x_j_list)
