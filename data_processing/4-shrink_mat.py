import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
import pickle
import math

# Parameters
epsilon = .01
directory_path = "C:/Users/bpiv4/Dropbox/CIS520/cis520/"

users_mat = sparse.load_npz(directory_path + 'data/user_mat.npz')
n = users_mat.shape[0]
d = users_mat.shape[1]
k = math.ceil(math.log(n/(epsilon**2), 2))
reduction_factor = math.sqrt(d)

random_proj = stats.rv_discrete(name='rand_proj', values=([-1, 0, 1],
 [1/(2*reduction_factor), (1-1/reduction_factor), 1/(2*reduction_factor)]))
 
proj_mat = math.sqrt(reduction_factor)*np.array(random_proj.rvs(size=d*k))
proj_mat = np.reshape(proj_mat, (d, k))
proj_mat = sparse.csc_matrix(proj_mat)

reduced_mat = 1/math.sqrt(k) * (users_mat * proj_mat)

open
