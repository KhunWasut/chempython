##### distance.py #####
# PART OF kpython.colvars

import numpy as np
import sys

from ..chemmatrixaux import index_verify_1d, atomindex_to_vecindex, vecindex_to_atomindex
from ..pbc import pair_dx


class ColvarDistanceException(Exception):
   pass


def dx_pbc(x_a_oneaxis, x_b_oneaxis, L):
   dx_oneaxis = x_a_oneaxis - x_b_oneaxis
   if (x_a_oneaxis - x_b_oneaxis) > L/2:
      dx_oneaxis -= L
   elif (x_a_oneaxis - x_b_oneaxis) < L/2:
      dx_oneaxis += L

   return dx_oneaxis


def dx_ab(X_m, vec_index_i, atom_a_index, atom_b_index, L):

   if index_verify_1d(vec_index_i, atom_a_index, atom_b_index) == 'NA':
      return 0
   elif index_verify_1d(vec_index_i, atom_a_index, atom_b_index) == 'FT':
      # Map vec_index_i's axis to atom b's axis. They need to be the same! 
      x_a_oneaxis = X_m[vec_index_i]
      atom_a_axis = vecindex_to_atomindex(vec_index_i)[1]
      x_b_oneaxis = X_m[atomindex_to_vecindex(atom_b_index, atom_a_axis)]
      
      return dx_pbc(x_a_oneaxis, x_b_oneaxis, L)
   elif index_verify_1d(vec_index_i, atom_a_index, atom_b_index) == 'LT':
      x_b_oneaxis = X_m[vec_index_i]
      atom_b_axis = vecindex_to_atomindex(vec_index_i)[1]
      x_a_oneaxis = X_m[atomindex_to_vecindex(atom_a_index, atom_b_axis)]

      return dx_pbc(x_a_oneaxis, x_b_oneaxis, L)*(-1.0)


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
      grad_x_rab.append(dx_ab(X_m, i, atom_a_index, atom_b_index, L)/(r_ab*2.0))

   return np.array(grad_x_rab)
