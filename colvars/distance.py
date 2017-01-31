##### distance.py #####

import numpy as np
import sys, os

from ..chemmatrixaux import atomindex_to_vecindex as atv
from ..chemmatrixaux import dx_pbc, d_dx_pbc

from ..pbc import pair_dx


def dr_dxi(x_a, x_b, a, b, i, L, r_ab):
   return d_dx_pbc(a, b, i, i)*dx_pbc(x_a, x_b, i, L)/r_ab


def d2r_dxjxi(x_a, x_b, a, b, i, j, L, r_ab):
   factor = d_dx_pbc(a, b, i, i)/(r_ab**2.0)
   hess_firstterm = r_ab * d_dx_pbc(a, b, i, j)
   hess_secondterm = dx_pbc(x_a, x_b, i, L) * dr_dxi(x_a, x_b, a, b, j, L, r_ab)

   return factor*(hess_firstterm - hess_secondterm)


# Constructing gradient vector grad_x(r_ab)
def grad_x(X_m, atom_a_index, atom_b_index, L):
   a = atom_a_index
   b = atom_b_index
   grad_x_list = []
   x_a = np.array([X_m[atv(a,'x')], X_m[atv(a,'y')], X_m[atv(a,'z')]])
   x_b = np.array([X_m[atv(b,'x')], X_m[atv(b,'y')], X_m[atv(b,'z')]])
   r_ab = pair_dx(x_a, x_b, L)

   for i in range(X_m.shape[0]):
      grad_x_list.append(dr_dxi(x_a, x_b, a, b, i, L, r_ab))

   return np.array(grad_x_list)


def hess_x_j(X_m, atom_a_index, atom_b_index, vec_index_j, L):
   a = atom_a_index
   b = atom_b_index
   j = vec_index_j

   hess_x_j_list = []

   x_a = np.array([X_m[atv(a,'x')], X_m[atv(a,'y')], X_m[atv(a,'z')]])
   x_b = np.array([X_m[atv(b,'x')], X_m[atv(b,'y')], X_m[atv(b,'z')]])
   r_ab = pair_dx(x_a, x_b, L)

   for i in range(X_m.shape):
      hess_x_j_list.append(d2r_dxjxi(x_a, x_b, a, b, i, j, L, r_ab))

   return np.array(hess_x_j_list)
