#!/usr/bin/env python3
import pickle
import numpy as np
import math
import scipy.stats
import scipy.sparse as sparse
from statistics import mean

# variables
FILE_PATH = "../data/"


def main():
    data = load_data('user_mat.npz')
    similarity = learn_all(data)
    similar_users = get_top_k(similarity[0], 2, 0)
    prediction = predict(data, similarity, similar_users, 1, 3)
    print(prediction)


def load_data(file_name):
    return sparse.load_npz(FILE_PATH + file_name)


def learn_all(data):
    # user by game
    num_users, num_games = np.shape(data)
    sim_pearson = np.zeros((num_users, num_users))

    for u1 in range(num_users):
        sim_pearson[u1] = learn_row(u1, data)

    return sim_pearson


def learn_row(row_inx, A):
    length, width = np.shape(A)
    data_1 = A[row_inx, :].A[0]
    arr = []
    for i in range(length):
        data_2 = A[i, :].A[0]
        r, p = scipy.stats.pearsonr(data_1, data_2)
        if math.isnan(r):
            r = 0
        arr.append(r)

    return arr


# returns indices of top k closest users in ascending order, excluding i (b/c same user )
def get_top_k(A, k, i):
    temp = np.copy(A)
    temp[i] = -2 # all other elements are [-1, 1]
    ind = np.argpartition(temp, -k)[-k:]
    return ind[np.argsort(temp[ind])[::-1]]


# indices of similar users
def predict(data, similarity, similar_users, user_inx, game_inx):
    other_predictions = []

    # find mean of prediction of this user
    mu = mean_prediction(data, user_inx)

    sum = 0
    for other_user in similar_users:
        # compute mean of prediction of this user
        m = mean_prediction(data, other_user)
        sum += similarity[user_inx, other_user] * (data[other_user, game_inx] - m)

    # want to predict
    denom = np.sum(np.absolute(similarity)[similar_users])

    return mu + sum / denom


def mean_prediction(data, user_inx):
    return mean(data[user_inx, :].A[0])


# run
main()
