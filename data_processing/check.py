import pickle
import numpy as np
import scipy.sparse as sparse
import math

directory_path = "C:/Users/bpiv4/Dropbox/CIS520/cis520/" # Path to git repo on your machine

users_mat = sparse.load_npz(directory_path + 'dense_user_mat.npz').todense()

sparsity = np.count_nonzero(users_mat) / (users_mat.shape[0] * users_mat.shape[1])
print(sparsity)
nonzero_counts = np.count_nonzero(users_mat, 1)
ordered_users = np.argsort(nonzero_counts, 0)[::-1]
nonzero_counts = nonzero_counts[ordered_users]

users_mat = np.squeeze(users_mat[ordered_users.flatten(),:])
nonzero_counts = np.count_nonzero(users_mat, 1)
for entry in nonzero_counts:
  print(entry)