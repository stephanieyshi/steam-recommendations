import numpy as np
import pickle
import math
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import sys

# Parameters
directory_path = "./../"
epochs = 50  # number of iterations over observations
threshold = 100  # convergence threshhold
eta = .008  # grad descent step size

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

# Global mean
mu = np.mean(entries)

users_mat.data = users_mat.data - mu

users_mat = users_mat.tocoo()

num_users = users_mat.shape[0]
num_games = users_mat.shape[1]


def bias_func(users_mat, user_bias, game_bias):
    total = 0
    for user, item, rui in zip(users_mat.row, users_mat.col, users_mat.data):
        # print(user_bias[0, user])
        # print(game_bias[0, item])
        # print(type(user_bias[0, user]))
        # print(type(game_bias[0, item]))
        curr = (rui - user_bias[0, user] - game_bias[0, item])**2
        total = total + curr

    # regularization
    # user_norm = np.linalg.norm(user_bias, 2)
    # item_norm = np.linalg.norm(item_bias, 2)
    # total = total + reg_param * (user_norm + item_norm)
    return total


print("Beginning prior")
user_bias = np.zeros((1, users_mat.shape[0]))
game_bias = np.zeros((1, users_mat.shape[1]))
# temp_dense = users_mat.todense()
# user_bias = np.divide(np.sum(temp_dense, axis=1),
#                       np.count_nonzero(temp_dense, axis=1))
# user_bias = np.squeeze(np.reshape(user_bias, (num_users, 1))).flatten()
# print(user_bias)
#
# game_bias = np.divide(np.sum(temp_dense, axis=0),
#                       np.count_nonzero(temp_dense, axis=0))
# game_bias = np.squeeze(np.reshape(game_bias, (num_games, 1))).flatten()
# print(game_bias)
#
# del temp_dense

# prior = np.zeros((num_users + num_games))
# prior[:num_users] = user_bias_prior
# prior[num_users:] = game_bias_prior

print("Intelligent prior done")
# options = {}
# options['maxfun'] = 3
# options['maxcor'] = 5000
# result = optimize.minimize(bias_func, prior,
#  (users_mat, num_users, num_games, 0), method='L-BFGS-B', options=options)
# user_bias = None
# game_bias = None
# if result.success:
#     fitted_params = result.x
#     user_bias = fitted_params[:num_users]
#     game_bias = fitted_params[num_users:]
#     print(user_bias)
#     print(game_bias)
# else:
#   print("Failure")

# Gradient descent attempt
print(game_bias.shape)
print(user_bias.shape)

curr_funct = bias_func(users_mat, user_bias, game_bias)
print("Objective function value: " +
      str(curr_funct))
prev_funct = 0
diff = 100000000
curr_epoch = 0
while diff > threshold and curr_epoch < epochs:
    # update_bi = np.zeros(game_bias.shape)
    # update_bu = np.zeros(user_bias.shape)
    for user, item, rui in zip(users_mat.row, users_mat.col, users_mat.data):
        bi_new = game_bias[0, item] + 2 * eta * \
            (rui - user_bias[0, user] - game_bias[0, item])
        bu_new = user_bias[0, user] + 2 * eta * \
            (rui - user_bias[0, user] - game_bias[0, item])
        user_bias[0, user] = bu_new
        game_bias[0, item] = bi_new
    #     update_bi[0, item] = update_bi[0, item] + 2 * eta * \
    #         (rui - user_bias[0, user] - game_bias[0, item])
    #     update_bu[0, user] = update_bu[0, user] - 2 * eta * \
    #         (rui - user_bias[0, user] - game_bias[0, item])
    # user_bias = update_bu + user_bias
    # game_bias = update_bi + game_bias
    curr_epoch = curr_epoch + 1
    prev_funct = curr_funct
    curr_funct = bias_func(users_mat, user_bias, game_bias)
    diff = prev_funct - curr_funct
    print("Objective function value: " +
          str(curr_funct))
    print("Objective function imporvement: " + str(diff))

outputs = (mu, user_bias, game_bias)
pickle_out = open(directory_path + "data/bias_exp.p", 'wb')
pickle.dump(outputs, pickle_out)
pickle_out.close()
