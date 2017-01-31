##### coordnum.py #####

import numpy as np
import sys, os

from ..chemmatrixaux import atomindex_to_vecindex as atv

from .distance import dr_dxi, d2r_dxjxi

from ..pbc import pair_dx

from .params import CNParams


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


# Calculate the first derivatives of both numerator and denominator
# The answers are simplified in terms of the values of num/denom themselves
# for ease of debug / readability
# dN/dxi = n(N-1)(dr/dxi)/r_ab
# dD/dxi = m(D-1)(dr/dxi)/r_ab
def dN_xi(n, r0, x_a, x_b, a, b, i, L, r_ab):
   N = n_ab_k(r_ab, r0, n)
   return (n*(N-1.0)*(dr_dxi(x_a, x_b, a, b, i, L, r_ab))/r_ab)


def dD_xi(m, r0, x_a, x_b, a, b, i, L, r_ab):
   D = d_ab_k(r_ab, r0, m)
   return (m*(D-1.0)*(dr_dxi(x_a, x_b, a, b, i, L, r_ab))/r_ab)


def grad_x(X_m, cn_params, L):
   # Simplify variable names
   a = cn_params.a_ind
   b_list = cn_params.b_inds
   n = cn_params.n
   m = cn_params.m
   r0 = cn_params.r0

   # n,d method aliases
   N = n_ab_k
   D = d_ab_k

   x_a = np.array([X_m[atv(a,'x')], X_m[atv(a,'y')], X_m[atv(a,'z')]])

   grad_x_list = []

   for i in range(X_m.shape[0]):
      sum_dcn_dxi = 0.0
      for b in b_list:
         x_b = np.array([X_m[atv(b,'x')], X_m[atv(b,'y')], X_m[atv(b,'z')]])
         r_ab = pair_dx(x_a, x_b, L)

         D_val = D(r_ab, r0, m)
         N_val = N(r_ab, r0, n)

         factor = D_val**(-2.0)
         grad_firstterm = D_val * dN_xi(n, r0, x_a, x_b, a, b, i, L, r_ab)
         grad_secondterm = N_val * dD_xi(m, r0, x_a, x_b, a, b, i, L, r_ab)

         sum_dcn_dxi += (factor*(grad_firstterm - grad_secondterm))

      grad_x_list.append(sum_dcn_dxi)

   return np.array(grad_x_list)


def hess_x_j(X_m, vec_index_j, cn_params, L):
   a = cn_params.a_ind
   b_list = cn_params.b_inds
   j = vec_index_j
   n = cn_params.numer_pow
   m = cn_params.denom_pow
   r0 = cn_params.r0

   N = n_ab_k
   D = d_ab_k

   x_a = np.array([X_m[atv(a,'x')], X_m[atv(a,'y')], X_m[atv(a,'z')]])

   hess_x_j_list = []

   for i in range(X_m.shape[0]):
      sum_d2cn_dxjxi = 0.0
      for b in b_list:
         x_b = np.array([X_m[atv(b,'x')], X_m[atv(b,'y')], X_m[atv(b,'z')]])
         r_ab = pair_dx(x_a, x_b, L)

         D_val = D(r_ab, r0, m)
         N_val = N(r_ab, r0, n)
         dDj = dD_xi(m, r0, x_a, x_b, a, b, j, L, r_ab)
         dNj = dN_xi(n, r0, x_a, x_b, a, b, j, L, r_ab)
         d2r_ji = d2r_xjxi(x_a, x_b, a, b, i, j, L, r_ab) 
         dr_i = dr_dxi(x_a, x_b, a, b, i, L, r_ab)
         dr_j = dr_dxi(x_a, x_b, a, b, j, L, r_ab)

         factor = D_val**(-4.0)
         hess_t1_1 = dr_i*(n*D_val*dNj)/r_ab
         hess_t1_2 = dr_i*(n*(N_val-1.0)*dDj)/r_ab
         hess_t1_3 = n*D_val*(N_val - 1.0)*d2r_ji/r_ab
         hess_t1_4 = (-1.0)*n*D_val*(N_val - 1.0)*dr_i*(dr_j/(r_ab*r_ab))
         hess_t1 = (D_val * D_val) * (hess_t1_1 + hess_t1_2 + hess_t1_3 + hess_t1_4)

         hess_t2_1 = dr_i*(m*N_val*dDj)/r_ab
         hess_t2_2 = dr_i*(m*(D_val-1.0)*dNj)/r_ab
         hess_t2_3 = m*N_val*(D_val - 1.0)*d2r_ji/r_ab
         hess_t2_4 = (-1.0)*m*N_val*(D_val - 1.0)*dr_i*(dr_j/(r_ab*r_ab))
         hess_t2 = (D_val * D_val) * (hess_t2_1 + hess_t2_2 + hess_t2_3 + hess_t2_4)

         hess_t3_1 = n*D_val*(N_val-1.0)*dr_i*dDj/r_ab
         hess_t3_2 = m*N_val*(D_val-1.0)*dr_i*dDj/r_ab
         hess_t3 = 2.0*D_val*(hess_t3_1 - hess_t3_2)

         sum_d2cn_dxjxi += (factor*(hess_t1 - hess_t2 - hess_t3))

      hess_x_j_list.append(sum_d2cn_dxjxi)

   return np.array(hess_x_j_list)
