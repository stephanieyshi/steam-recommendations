import numpy as np
import pickle
import math
import seaborn as sns
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
directory_path = "C:/Users/bpiv4/Dropbox/CIS520/cis520/"

with open(directory_path + 'data/games.p', 'rb') as f:
    games = pickle.load(f)
    f.close()

users_map = {}
with open(directory_path + 'data/user_map.p', 'rb') as f:
    users_map = pickle.load(f)
    f.close()

users = {}
with open(directory_path + 'data/train_users_08.p', 'rb') as f:
    users = pickle.load(f)
    f.close()

users_mat = sparse.load_npz(directory_path + 'data/user_mat.npz')
entries = users_mat.data

sns.distplot(entries)
plt.show()
