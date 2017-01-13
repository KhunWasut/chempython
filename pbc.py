# PBC Module

# Imports
import numpy as np

# Exceptions
class PBCException(Exception):
   pass

# Objects


# Methods
def pair_dx(c1,c2,L):
   """
   pair_dx(c1,c2,L)
   
   Calculates a euclidean distance between any pair of atoms in a.u.

   Input
   ----------
   c1: A coordinate array of atom 1 (1D numpy array size 3)
   c2: A coordinate array of atom 2 with the structure (1D numpy array size 3)
   L: A cubic box's box length in a.u.

   Output
   ----------
   euclidean_distance(dxyz):
       Determined from an inner method which returns a non-negative float.

   """

   # Check if c1 and c2 are np.ndarray
   try:
      if (not isinstance(c1, np.ndarray)) or (not isinstance(c2, np.ndarray)):
         raise PBCException

   except PBCException:
      print('PBCException: Either c1 or c2 is not a numpy array. Check your code!')
      sys.exit(1)

   # The condition must check that len(c1) = len(c2)
   try:
      if c1.shape != c2.shape:
         raise PBCException

      # NumPy implementation
      dxyz = c2 - c1    # Vector subtraction

      # Array masks
      # The edited value should fall in desirable range so we dont need
      # to use a temp array and the following two lines will never overwrite.
      dxyz[dxyz < (-L/2.0)] += L
      dxyz[dxyz > (L/2.0)] -= L

      return np.sqrt(np.sum(dxyz**2))
     
   except PBCException:
      print('PBCException: Coordinate array lengthes mismatch. Exiting...')
      sys.exit(1)
