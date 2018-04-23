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

train_users = {}
with open(directory_path + 'data/train_users_04.p', 'rb') as f:
    users = pickle.load(f)
    f.close()

test_users = {}
with open(directory_path + 'data/test_users_04.p', 'rb') as f:
    users = pickle.load(f)
    f.close()

for user, game_dict in train_users.items():
    for game, hours in game_dict.items():
        if hours > 6:
            print('stop')
        if hours < 0:
            print('stop')
for user, game_dict in test_users.items():
    for game, hours in game_dict.items():
        if hours > 6:
            print('stop')
        if hours < 0:
            print('stop')
