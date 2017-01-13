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
\

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
