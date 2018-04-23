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

# Parameters
it = 25    # number of iterat ions
epochs = 20  # epochs for each model
target_name = '04'  # density of the data
subset_size = .7  # size of subset for each  as a fraction of total data
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

weights = np.ones(len(train_list))
weight_inds = np.arange(start=0, stop=np.size(weights))
subset_size = math.ceil(subset_size * len(weights))
# functions


def write_file(subset, file_name):
    with open(file_name, 'w') as f:
        for row in subset:
            user = row[0]
            game = row[1]
            hours = train_users[user][game][0]
            # print(hours)
            f.write('%s\t%s\t%d\n' % (user, game, hours))
    f.close()


beta = np.zeros(it)
av_loss = 0
# NOTE change rating scale
reader = Reader(line_format='user item rating',
                sep='\t', rating_scale=(0, 6))

print('Setup complete')
for i in range(it):
    print('Preparing to train model ' + str(i) + '...')
    model_path = base_model_path + str(i) + '_' + target_name + '.p'
    probs = np.divide(weights, np.sum(weights))
    subset_inds = np.random.choice(
        weight_inds, subset_size, replace=True, p=probs)
    # print(subset_inds.shape)
    subset = train_list[subset_inds]
    # print(subset.shape)
    write_file(subset, file_path)
    del subset
    del subset_inds

    train_data = Dataset.load_from_file(
        file_path, reader).build_full_trainset()

    print('Training model ' + str(i) + '...')
    algo = SVD(verbose=True, n_factors=factors, n_epochs=epochs,
               reg_bu=user_reg, reg_bi=game_reg,
               reg_pu=user_vec_reg, reg_qi=game_vec_reg)
    algo.fit(train_data)

    print('Done Training Model ' + str(i))
    print('Saving model...')
    predictions = algo.test(train_data.build_testset(), verbose=False)
    dump.dump(file_name=model_path, algo=algo, verbose=True)
    del algo
    print('Updating weights...')
    subset_inds = np.zeros(len(predictions))
    est_vec = np.zeros(len(predictions))
    true_vec = np.zeros(len(predictions))
    count = 0
    subset_inds = []
    for user, game, r_ui, est, _ in predictions:
        est_vec[count] = est
        true_vec[count] = r_ui
        subset_inds.append(train_users[user][game][1])
        count += 1

    # calculate loss
    loss = np.square(np.subtract(est_vec, true_vec))
    d = np.amax(loss)
    loss = np.divide(loss, d)

    # calculate average loss and inverse confidence
    av_loss = np.sum(np.multiply(loss, probs[subset_inds]))
    beta[i] = av_loss/(1-av_loss)
    weights[subset_inds] = weights[subset_inds] * np.power(beta[i], 1-loss)
