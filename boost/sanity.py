import numpy as np
import pickle
import math
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import sys
from surprise.model_selection import cross_validate
from collections import defaultdict
import pickle
import os
import random
from surprise import dump
from surprise import SVD, Reader, Dataset, KNNBasic, KNNWithMeans, \
    NormalPredictor, NMF, accuracy

factors = 100  # number of factors
user_reg = .02
game_reg = .02
user_vec_reg = .02
game_vec_reg = .02

directory_path = 'C:/Users/bpiv4/Dropbox/CIS520/cis520/'
file_path = directory_path + 'data/curr_svd_data.csv'
base_model_path = directory_path + 'data/boost_model'
# path to training dictionary
train_path = directory_path + 'data/train_users_' + target_name + '.p'
test_path = directory_path + 'data/test_users_' + target_name + '.p'

train_users = {}
with open(train_path, 'rb') as f:
    train_users = pickle.load(f)
    f.close()

test_users = {}
with open(test_path, 'rb') as f:
    test_users = pickle.load(f)
    f.close()

# associate users with weights
train_list = []
ind = 0
for user, game_dict in train_users.items():
    for game, hours in game_dict.items():
        train_list.append((user, game))
        if hours > 6:
            print("FUCK")
        elif hours < 0:
            print("DICK")
        game_dict[game] = (float(hours), ind)
        ind = ind + 1
train_list = np.array(train_list, dtype=('str', 'str'))
