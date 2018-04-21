import numpy as np
import pickle
import math
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# Parameters
directory_path = "C:/Users/bpiv4/Dropbox/CIS520/cis520/"  # path to directory
# name of file containing global mean, user bias, and item bias
type_name = "train"  # train or test
size_name = "04"  # name of data density
shrinkage = 100  # Beta in shrinkage formula

mat_name = type_name + "_user_mat_" + size_name + ".npz"  # name of user matrix
pearson_name = type_name + "_pearson_" + \
    size_name + ".p"  # name of pearson matrix
shrunk_name = type_name + "_shrunk_pearson_" + size_name + ".p"

user_mat = sparse.load_npz(directory_path + "data/" + mat_name)
num_games = user_mat.size[1]
pearson_mat = None

with open(directory_path + 'data/' + pearson_name, 'rb') as f:
    pearson_mat = pickle.load(f)
    f.close()

pearson_mat = np.tril(pearson_mat)
for i in range(num_games):
    for j in range(i):
        curr = pearson_mat[i, j]
        arr_1 = user_mat[:, i].toarray()
        arr_2 = user_mat[:, j].toarray()
        num_shared = np.count_nonzero(np.logical_and(arr_1 > 0, arr_2 > 0))
        pearson_mat[i, j] = num_shared / (num_shared + shrinkage) * curr

pickle_out = open(directory_path + "data/" + shrunk_name, 'wb')
pickle.dump(pearson_mat, pickle_out)
pickle_out.close()
