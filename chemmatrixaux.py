##### chemmatrixaux.py #####

# Maps or de-maps the atom number to the 3N index vector
# X = [x_1, x_2, ..., x_3N], where (x_(3(l-1)), x_(3(l-1)+1), x_(3(l-1)+2)) are coordinates corresponding to atom l

import numpy as np
import sys


class ChemMatrixAuxException(Exception):
   pass


# Axis to number mapping dictionaries
axis_to_number = {'x': 0, 'y': 1, 'z': 2}
number_to_axis = {0: 'x', 1: 'y', 2: 'z'}


# Mapper
def atomindex_to_vecindex(atom_a_index, axis):
   return (3 * (atom_a_index) + axis_to_number['axis'])


def vecindex_to_atomindex(vec_index_i):
   atom_a_index = int(vec_index_i / 3)
   axis_number = vec_index_i % 3
   return (atom_a_index, number_to_axis[axis_number])


# Indices verification
def index_verify_1d(vec_index_i, atom_a_index, atom_b_index):
   # Returns codes 'NA', 'FT', 'LT'
   if atom_a_index == atom_b_index:
      raise ChemMatrixAuxException

   try:
      the_atom = vecindex_to_atomindex(vec_index_i)

      if the_atom[0] == atom_a_index:
         return 'FT'
      elif the_atom[0] == atom_b_index:
         return 'LT'
      else:
         return 'NA'
      
   except ChemMatrixAuxException:
      print('ChemMatrixAuxException: index_verify_1d: atom_a and atom_b cannot be the same atom!')
      sys.exit(1)


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


def d_dx_ab(vec_index_i, atom_a_index, abom_b_index):
   # Works similar to dx_ab, but this will mainly be used with second derivatives where 
   # the terms that do not relate to atom a or b in the expression goes to zero.

   if index_verify_1d(vec_index_i, atom_a_index, atom_b_index) == 'NA':
      return 0.0
   elif index_verify_1d(vec_index_i, atom_a_index, atom_b_index) == 'FT':
      return 1.0
   elif index_verify_1d(vec_index_i, atom_a_index, atom_b_index) == 'LT':
      return -1.0

