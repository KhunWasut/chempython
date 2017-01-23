### necessary imports ###

import kpython_path as kp

import os, re


# Read necessary data from our workspace scheme
x_snapshot_filelist = os.listdir('./x-snapshots')
f_snapshot_filelist = os.listdir('./f-snapshots')

# Assert equal length
assert(len(x_snapshot_filelist) == len(f_snapshot_filelist))

# Actually read the vectors
X = []
GRAD_V = []

for i in range(len(x_snapshot_filelist)):
    path_x = os.path.join('./x-snapshots', x_snapshot_filelist[i])
    path_f = os.path.join('./f-snapshots', f_snapshot_filelist[i])

    vec_x = kp.chemmatrix.readvec(path_x)
    vec_f = kp.chemmatrix.readvec(path_f)

    X.append(vec_x)
    GRAD_V.append(vec_f)

##### END OF PART 1 - MAKE SURE THIS PART WORKS BEFORE PROCEEDING! #####

atom_vec = kp.chemmatrix.readvec('./aux/atoms.vec')

masses = []
# Build a mass matrix
for atom in atom_vec:
    if atom in kp.constants.atom_masses.keys:
        masses.append(kp.constants.atom_mass[key])

mass_vec = np.array(masses)
mu = np.diag(mass_vec)

##### END OF PART 2 - MAKE SURE THIS PART WORKS BEFORE PROCEEDING! #####

icf = []
CN = kp.colvars.coordnum.CNParams(numer_pow=10.0, denom_pow=26.0, r0=3.2)
R = kp.colvars.distance.RParams()
L = 11.093

# Before calculations, make sure that distance c.v. is in atomic units!

for xi, grad_vi in X, GRAD_V:
    icf_i = kp.icf.icf_construct(xi, mu, grad_vi, kT, R, CN, L)
    for icf_val in icf_i:
        icf.append(icf_val)

icf = np.array(icf)

##### END OF PART 3 - MAKE SURE THIS PART WORKS BEFORE PROCEEDING! #####

icf_var = kp.icf.var(icf, num_cv=2)

### Actually calculate CN and construct r_vec and cn_vec vectors
### Using what we have

##### END OF PART 4 - MAKE SURE THIS PART WORKS BEFORE PROCEEDING! #####

r_min = 1.5
r_max = 7.0
cn_min = 2.5
cn_max = 5.5

theta_r = 6.0
theta_cn = 4.0

chi = 10.0  # kcal/mol

r_space = np.linspace(r_min, r_max, 1000)
cn_space = np.linspace(cn_min, cn_max, 1000)

fe_coord = []

for r in r_space:
    for cn in cn_space:
        xi_star_vec = np.array([r, cn])
        fe = kp.fe_construct.fe_fit(r_vec, cn_vec, theta_r, theta_cn, chi, icf, icf_var, xi_star_vec)
        fe_coord.append(fe)

##### END OF PART 5 - MAKE SURE THIS PART WORKS BEFORE PROCEEDING! #####

# Use matplotlib here to plot fe_coord or save to a file with pandas

##### END OF PART 6 - MAKE SURE THIS PART WORKS BEFORE PROCEEDING! #####

##### END OF SCRIPT #####
