#### chemmatrix.py ####

import numpy as np
import pandas as pd


# This module reads matrices / vector from standard .vec files
def read_vec(file_path):
    # Implement check for .vec file in the future!
    df = pd.read_csv(file_path, sep='\s+', header=None)
    vec = df.as_matrix()
    # vec is a 2D array with 1 column. We want only 1D array!
    vec = vec.reshape(vec.shape[0])
    return vec
