##### distance.py #####
# PART OF kpython.colvars

import numpy as np
import sys

from ..chemmatrixaux import index_verify_1d, atomindex_to_vecindex, vecindex_to_atomindex, dx_pbc, dx_ab
from ..pbc import pair_dx


class ColvarDistanceException(Exception):
   pass


def first_deriv_rab(X_m, atom_a_index, atom_b_index, sys_params):
   # INPUT PARAMETERS:
   #    X_m: m-th datapoint's coordinates (3N x 1 size)
   #    atom_a_index: The atom index of atom a in r_ab
   #    atom_b_index: The atom index of atom b in r_ab
   # OUTPUT PARAMETER:
   #    grad_x(r_ab): Gradient vector of r_ab wrt all x in X_m (3N x 1 size)
   #    NOTE: If there are D colvars in system, we construct grad matrix in main program
   #          by augmenting all gradient vectors of each individual colvar.
   
   try:
      L = sys_params['L']
   except KeyError:
      print('KeyError: In first_deriv_rab: sys_params contains no box size \'L\'.')
      sys.exit(1)

   x_a_x_index = atomindex_to_vecindex(atom_a_index, 'x'.lower())
   x_a_y_index = atomindex_to_vecindex(atom_a_index, 'y'.lower())
   x_a_z_index = atomindex_to_vecindex(atom_a_index, 'z'.lower())
   x_b_x_index = atomindex_to_vecindex(atom_b_index, 'x'.lower())
   x_b_y_index = atomindex_to_vecindex(atom_b_index, 'y'.lower())
   x_b_z_index = atomindex_to_vecindex(atom_b_index, 'z'.lower())

   x_a = np.array([X_m[x_a_x_index], X_m[x_a_y_index], X_m[x_a_z_index]])
   x_b = np.array([X_m[x_b_x_index], X_m[x_b_y_index], X_m[x_b_z_index]])
   r_ab = pair_dx(x_a, x_b, L)

   grad_x_rab = []

   for i in range(X_m.shape):
      grad_x_rab.append(dx_ab(X_m, i, atom_a_index, atom_b_index, L)/(r_ab))

   return np.array(grad_x_rab)


def second_deriv_rab_wrt_xj(X_m, vec_index_j, atom_a_index, atom_b_index, sys_params):
   # INPUT PARAMETERS:
   #    X_m: m-th datapoint's coordinates (3N x 1 size)
   #    atom_a_index: The atom index of atom a in r_ab
   #    atom_b_index: The atom index of atom b in r_ab
   #    vec_index_j: The j-th coordinate vector index for the derivative with respect to xj
   # OUTPUT PARAMETER:
   #    hess_xj(r_ab): j-th column vector of the Hessian matrix of r_ab wrt all x in X_m (3N x 1 size)

   try:
      L = sys_params['L']
   except KeyError:
      print('KeyError: In second_deriv_rab_wrt_xj: sys_params contains no box size \'L\'.')
      sys.exit(1)

   x_a_x_index = atomindex_to_vecindex(atom_a_index, 'x'.lower())
   x_a_y_index = atomindex_to_vecindex(atom_a_index, 'y'.lower())
   x_a_z_index = atomindex_to_vecindex(atom_a_index, 'z'.lower())
   x_b_x_index = atomindex_to_vecindex(atom_b_index, 'x'.lower())
   x_b_y_index = atomindex_to_vecindex(atom_b_index, 'y'.lower())
   x_b_z_index = atomindex_to_vecindex(atom_b_index, 'z'.lower())

   x_a = np.array([X_m[x_a_x_index], X_m[x_a_y_index], X_m[x_a_z_index]])
   x_b = np.array([X_m[x_b_x_index], X_m[x_b_y_index], X_m[x_b_z_index]])
   r_ab = pair_dx(x_a, x_b, L)

   # TO BE CONTINUED...
   # Build a list of domain indices for r_ab
   possible_domains = [x_a_x_index, x_a_y_index, x_a_z_index, x_b_x_index, x_b_y_index, x_b_z_index]

   # If j is not in these domain, return zero vector
   if not (vec_index_j in possible_domains):
      return np.zeros(X_m.shape)
   else:
      hess_xj_rab = []
      for k in range(X_m.shape):
         # Determine where k belongs
         the_atom = vecindex_to_atomindex(k)

         if the_atom[0] == atom_a_index:
            main_sign = 1.0
            another_numer_atom_vec_index = atomindex_to_vecindex(atom_b_index, the_atom[1])
            numer_domains = [k, another_numer_atom_vec_index]

            if not (vec_index_j in numer_domains):
               d_dx_by_dxj = 0.0
            else:
               if vec_index_j == numer_domains[0]:
                  d_dx_by_dxj = 1.0
               elif vec_index_j == numer_domains[1]:
                  d_dx_by_dxj = -1.0

            dx = dx_pbc(X_m[numer_domains[0]], X_m[numer_domains[1]], L)
            d_rab_dy_dxj = dx_ab(X_m, vec_index_j, atom_a_index, atom_b_index, L)/(r_ab)

            hess = main_sign*((r_ab*d_dx_by_dxj)-(dx*d_rab_by_dxj))/(r_ab*r_ab)
         elif the_atom[0] == atom_b_index:
            main_sign = -1.0
            another_numer_atom_vec_index = atomindex_to_vecindex(atom_a_index, the_atom[1])
            numer_domains = [k, another_numer_atom_vec_index]

            if not (vec_index_j in numer_domains):
               d_dx_by_dxj = 0.0
            else:
               if vec_index_j == numer_domains[0]:
                  d_dx_by_dxj = -1.0
               elif vec_index_j == numer_domains[1]:
                  d_dx_by_dxj = 1.0

            dx = dx_pbc(X_m[numer_domains[1]], X_m[numer_domains[0]], L)
            d_rab_dy_dxj = dx_ab(X_m, vec_index_j, atom_a_index, atom_b_index, L)/(r_ab)

            hess = main_sign*((r_ab*d_dx_by_dxj)-(dx*d_rab_by_dxj))/(r_ab*r_ab)

         else:    # Does not belong at all, simply append 0 to the entry
            hess_xj_rab.append(0.0)

      return np.array(hess_xj_rab)
