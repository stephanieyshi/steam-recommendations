import numpy as np
import pickle
import math
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import random

# Parameter size
directory_path = './../'
data_path = directory_path + 'data/'
subset_size = 50000
k = 10

# Paths
train_mat_path = data_path + 'train_user_mat_08.npz'
bias_path = data_path + 'train_bias_08.p'

bias = []
with open(bias_path, 'rb') as f:
    bias = pickle.load(f)
    f.close()
mu = bias[0]
buvec = bias[1]
bivec = bias[2]

# path to directory
with open(data_path + 'train_shrunk_pearson_08.p', 'rb') as f:
    pearson = pickle.load(f)
    pearson = pearson + pearson.transpose()
f.close()

user_mat = sparse.load_npz(train_mat_path)

wij = np.zeros(pearson.shape) + 0.01

with open(data_directory + 'qi.p', 'rb') as f:
    q = pickle.load(f)
f.close()

# path to directory
with open(data_directory + 'pu.p', 'rb') as f:
    p = pickle.load(f)
f.close()

output = {}
output['bu'] = buvec
output['bi'] = bivec
output['pearson'] = pearson
output['mu'] = mu
output['p'] = p
output['q'] = q
output['wij'] = wij

pickle_out = open(directory_path + 'data/ensemble_init_08.p', 'wb')
pickle.dump(output, pickle_out)
pickle_out.close()
