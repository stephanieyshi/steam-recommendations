import numpy as np
import scipy.sparse as sparse
import pickle

# Parameters
epsilon = .01
users_mat = sparse.load_npz('./data/users_mat.npz')
n = users_mat.shape[0]
games = users_mat.shape[1]
k = math.ceil(math.log(n/(epsilon^2), 2)

