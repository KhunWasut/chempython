# cn.py
# Calculates coordination number. Assume standard unit similar to NWChem (a.u.)

# imports
from .pbc import pair_dx
import numpy as np

# exceptions
class CNException(Exception):
   pass


# objects


# methods
def cn_onetoall(group1, group2_array, nom_pow, denom_pow, r0, L):
   """
   cn_onetoall(group1, group2_array, nom_pow, denom_pow, r0)

   Calculates coordination number when group1 has one atom and group2 has multiple atoms


   input
   ----------
   group1 - Base atom of the format (1D numpy array size 3)
   group2_array - Essentially a list of atoms (2D numpy array size 3*N)
   nom_pow - nominator's factor
   denom_pow - denominator's factor
   r0 - r0 in a.u.
   L - box size in a.u. 

   output
   ----------
   cn - The calculated coordination number
   """

   # A list that saves all pair distances
   d = []

   # Loop over number of rows in group2_array
   for i in range(group2_array.shape[0]):
      d_indiv = pair_dx(group1, group2_array[i,:], L)
      d.append(d_indiv)

   # Turn d into np.ndarray
   d = np.array(d)

   # Array operation for calculating the individual CN
   # n_ij = (1-(r_ij/r0)**n)/(1-(r_ij/r0)**m)
   n = (1.0 - (d / r0)**nom_pow) / (1.0 - (d / r0)**denom_pow)

   # The total coordination number is simply the sum of all values in n
   return np.sum(n)
