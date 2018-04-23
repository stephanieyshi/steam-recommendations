import numpy as np
import pickle
import math
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import random

directory_path = './../'
data_path = directory_path + 'data/'

# Paths
output_path = data_path + 'ensemble_dump_08.p'
test_users_path = data_path + 'test_users_08.p'
games_map_path = data_path + 'train_games_08.p'
user_map_path = data_path + 'train_user_map_08.p'
train_mat_path = data_path + 'train_user_mat_08.p'

# Imports
output = {}
with open(output_path, 'rb') as f:
    output = pickle.load(f)
    f.close()


buvec = output['bu']
bivec = output['bi']
pearson = output['pearson']
mu = output['mu']
p = output['p']
q = output['q']
wij = output['wij']

del output

user_map = {}
with open(user_map_path, 'rb') as f:
    user_map = pickle.load(f)
    f.close()

games_map = {}
with open(games_map_path, 'rb') as f:
    games_map = pickle.load(f)
    f.close()

train_mat = sparse.load_npz(train_mat_path)
