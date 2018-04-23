import numpy as np
import pickle
import math
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# Parameters
directory_path = "./../"

with open(directory_path + 'data/train_games_08.p', 'rb') as f:
    games = pickle.load(f)
    f.close()

users_map = {}
with open(directory_path + 'data/train_user_map_08.p', 'rb') as f:
    users_map = pickle.load(f)
    f.close()

users = {}
with open(directory_path + 'data/train_users_08.p', 'rb') as f:
    users = pickle.load(f)
    f.close()

users_mat = sparse.load_npz(directory_path + 'data/train_user_mat_08.npz')
entries = users_mat.data

num_games = users_mat.shape[1]

pearson_mat = np.zeros((num_games, num_games))
count = 0
for i in range(num_games):
    for j in range(i):
        pearson_mat[i, j] = stats.pearsonr(
            users_mat[:, i].toarray(), users_mat[:, j].toarray())[0]
        count = count + 1
        if count % 10000 == 0:
            print(count)

pearson_mat = pearson_mat + np.transpose(pearson_mat)

pickle_out = open(directory_path + "data/train_pearson_08.p", 'wb')
pickle.dump(pearson_mat, pickle_out)
pickle_out.close()
