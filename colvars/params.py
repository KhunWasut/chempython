##### colvars/params.py #####

# DATA STRUCTURES FOR COLVARS


class RParams:
    a_ind = 0
    b_ind = 0

    def __init__(self, a=0, b=0):
        self.a_ind = a
        self.b_ind = b


class CNParams:
    # Standard unit in this data structure: angstroms
    a_ind = 0
    b_inds = []
    n = 0
    m = 0
    r0 = 0

    def __init__(self, a=0, b_list=[], n=0, m=0, r0=0):
        self.a_ind = a
        self.b_inds = b_list
        self.n = n
        self.m = m
        self.r0 = r0
