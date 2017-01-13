##### coordnum.py #####
# PART OF kpython.colvars module

import numpy as np
import sys

from ..chemmatrixaux import index_verify_1d, vecindex_to_atomindex, atomindex_to_vecindex, dx_pbc, dx_ab
from ..pbc import pair_dx


# Data structure
class CNParams():
   def __init__(self, numer_pow=6.0, denom_pow=12.0, r0=2.0):
      self.numer_pow = numer_pow
      self.denom_pow = denom_pow
      self.r0 = r0


def n_ab(r_ab, r0, numer_pow):
   return (1.0-((r_ab)/(r0))**(numer_pow))


def d_ab(r_ab, r0, denom_pow):
   return (1.0-((r_ab)/(r0))**(denom_pow))


def dn_ab_dxi(r_ab, r0, numer_pow, vec_index_i, atom_a_index, atom_b_index, L):
   # Need to come back and assert that r_ab corresponds to vec_index_i
   dx = dx_ab(vec_index_i, atom_a_index, atom_b_index, L)
   return (-1.0)*dx*(r_ab**(numer_pow-2))*(numer_pow)/(r0**numer_pow)


def dd_ab_dxi(r_ab, r0, denom_pow, vec_index_i, atom_a_index, atom_b_index, L):
   # Need to come back and assert that r_ab corresponds to vec_index_i
   dx = dx_ab(vec_index_i, atom_a_index, atom_b_index, L)
   return (-1.0)*dx*(r_ab**(denom_pow-2))*(denom_pow)/(r0**denom_pow)


def first_deriv_cn(X_m, atom_a_index, atom_b_list_of_indices, sys_params, mg_o_cn_params):
   # INPUT PARAMETERS:
   #    X_m: m-th datapoint's coordinates (3N x 1 size)
   #    atom_a_index: atom a's index (1 atom)
   #    atom_b_list_of_indices: list of indices for atom b (n atoms)
   #    mg_o_cn_params: The CN parameter object
   # OUTPUT PARAMETER:
   #    grad_x(cn): Gradient vector of CN wrt all x in X_m (3N x 1 size)
   
   try:
      L = sys_params['L']
   except KeyError:
      print('KeyError: In second_deriv_rab_wrt_xj: sys_params contains no box size \'L\'.')
      sys.exit(1)

   grad_x_cn = []

   for vec_index_i in range(X_m.shape):
      sum_grad = 0.0
      for atom_b_index in atom_b_list_of_indices:
         x_a_x_index = atomindex_to_vecindex(atom_a_index, 'x'.lower())
         x_a_y_index = atomindex_to_vecindex(atom_a_index, 'y'.lower())
         x_a_z_index = atomindex_to_vecindex(atom_a_index, 'z'.lower())
         x_b_x_index = atomindex_to_vecindex(atom_b_index, 'x'.lower())
         x_b_y_index = atomindex_to_vecindex(atom_b_index, 'y'.lower())
         x_b_z_index = atomindex_to_vecindex(atom_b_index, 'z'.lower())

         x_a = np.array([X_m[x_a_x_index], X_m[x_a_y_index], X_m[x_a_z_index]])
         x_b = np.array([X_m[x_b_x_index], X_m[x_b_y_index], X_m[x_b_z_index]])
         r_ab = pair_dx(x_a, x_b, L)

         n_ab_thisterm = n_ab(r_ab, mg_o_cn_params.r0, mg_o_cn_params.numer_pow)
         d_ab_thisterm = d_ab(r_ab, mg_o_cn_params.r0, mg_o_cn_params.denom_pow)
         dn_ab_dxi_thisterm = dn_ab_dxi(r_ab, mg_o_cn_params.r0, mg_o_cn_params.numer_pow, vec_index_i, atom_a_index, atom_b_index, L)
         dd_ab_dxi_thisterm = dd_ab_dxi(r_ab, mg_o_cn_params.r0, mg_o_cn_params.denom_pow, vec_index_i, atom_a_index, atom_b_index, L)

         addition = (d_ab_thisterm*dn_ab_dxi_thisterm - n_ab_thisterm*dd_ab_dxi_thisterm)/(d_ab_thisterm**2.0)
         sum_grad += addition

      grad_x_cn.append(sum_grad)

   return np.array(grad_x_cn)


def second_deriv_cn_wrt_xj(X_m, vec_index_j, atom_a_index, atom_b_list_of_indices, sys_params, mg_o_cn_params):

   try:
      L = sys_params['L']
   except KeyError:
      print('KeyError: In second_deriv_rab_wrt_xj: sys_params contains no box size \'L\'.')
      sys.exit(1)

   hess_xj_cn = []

   for vec_index_i in range(X_m.shape):
      sum_grad = 0.0
      for atom_b_index in atom_b_list_of_indices:
         x_a_x_index = atomindex_to_vecindex(atom_a_index, 'x'.lower())
         x_a_y_index = atomindex_to_vecindex(atom_a_index, 'y'.lower())
         x_a_z_index = atomindex_to_vecindex(atom_a_index, 'z'.lower())
         x_b_x_index = atomindex_to_vecindex(atom_b_index, 'x'.lower())
         x_b_y_index = atomindex_to_vecindex(atom_b_index, 'y'.lower())
         x_b_z_index = atomindex_to_vecindex(atom_b_index, 'z'.lower())

         x_a = np.array([X_m[x_a_x_index], X_m[x_a_y_index], X_m[x_a_z_index]])
         x_b = np.array([X_m[x_b_x_index], X_m[x_b_y_index], X_m[x_b_z_index]])
         r_ab = pair_dx(x_a, x_b, L)

         possible_domains = [x_a_x_index, x_a_y_index, x_a_z_index, x_b_x_index, x_b_y_index, x_b_z_index]
