import re


# Mass Dictionary (in a.m.u.)
# for now, only list those relevant to our simulation
masses = {
    'H': 1.0079,
    'Li': 6.941,
    'O': 15.9994,
    'Cl': 35.453
}


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]
