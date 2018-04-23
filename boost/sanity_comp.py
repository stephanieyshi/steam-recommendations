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
import re

directory = "C:/Users/bpiv4/Dropbox/CIS520/cis520/"
bran_csv_path = directory + 'boost/bran.csv'
bar_csv_path = directory + 'boost/bar.csv'

bar_users = {}
bran_users = {}
bran_total = 0
bar_total = 0
with open(bran_csv_path, 'rb') as f:
    for line in f:
        line = line.decode('utf-8')
        line = line.strip()
        line = re.split('\t+', line)
        user = line[0].strip()
        game = line[1].strip()
        hours = line[2].strip()
        bran_total = bran_total + 1
        if user not in bran_users:
            game_dict = {}
            game_dict[game] = hours
            bran_users[user] = game_dict
        else:
            game_dict = bran_users[user]
            game_dict[game] = hours
            bran_users[user] = game_dict
    f.close()

with open(bar_csv_path, 'rb') as f:
    for line in f:
        line = line.decode('utf-8')
        line = line.strip()
        line = re.split('\t+', line)
        user = line[0].strip()
        game = line[1].strip()
        hours = line[2].strip()
        bar_total = bar_total + 1
        if user not in bar_users:
            game_dict = {}
            game_dict[game] = hours
            bar_users[user] = game_dict
        else:
            game_dict = bar_users[user]
            game_dict[game] = hours
            bar_users[user] = game_dict
    f.close()

print('Bran Total: ' + str(bran_total))
print('Bar Total: ' + str(bar_total))

for user, game_dict in bar_users.items():
    for game, hours in game_dict.items():
        game_dict[game] = float(hours)

for user, game_dict in bran_users.items():
    for game, hours in game_dict.items():
        game_dict[game] = float(hours)

for user, game_dict in bar_users.items():
    bran_dict = bran_users[user]
    for game, hours in game_dict.items():
        bran_hours = bran_dict[game]
        if hours != bran_hours:
            print(user)
            print(game)
            print('Bar: ' + str(hours))
            print('Bran: ' + str(bran_hours))
