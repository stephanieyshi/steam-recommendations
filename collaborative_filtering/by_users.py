#!/usr/bin/env python3
import pickle
import numpy as np
import math
import scipy.stats
import scipy.sparse as sparse

# variablesp
FILE_PATH = "../data/"


def main():
    data = load_data('user_mat.npz')
    similarity = learn_all(data)
    print(similarity[0])
    inx = get_top_k(similarity[0], 2)
    print('INDICES')
    print(inx)
    print('DONE')


def load_data(file_name):
    print('LOADING DATA')
    return sparse.load_npz(FILE_PATH + file_name)


def learn_all(data):
    # user by game
    num_users, num_games = np.shape(data)
    sim_pearson = np.zeros((num_users, num_users))

    for u1 in range(num_users):
        sim_pearson[u1] = learn_row(u1, data, num_users)

    return sim_pearson


def learn_row(row_inx, A, length):
    data_1 = A[row_inx, :].A[0]
    arr = []
    for i in range(length):
        data_2 = A[i, :].A[0]
        r, p = scipy.stats.pearsonr(data_1, data_2)
        if math.isnan(r):
            r = 0
        arr.append(r)

    return arr


def get_top_k(A, k):
    return np.argpartition(A, -k)


def predict():
    print('PREDICTING')


def evaluate():
    print('hello')
    # TODO: calculate MAE
    # TODO: calculate RMSE


# run
main()
