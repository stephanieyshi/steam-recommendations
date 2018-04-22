#!/usr/bin/env python3
import numpy as np
import math
import scipy.stats
import scipy.sparse as sparse
from statistics import mean
from sklearn.metrics import mean_squared_error

# variables
FILE_PATH = "../data/"


def main():
    k = 5
    user_inx = 0
    game_inx = 3
    data = load_data('small_user_mat.npz')
    num_users, num_games = np.shape(data)

    models = ['pearson', 'cosine', 'jaccard']

    for model in models:
        similarity = learn_all(data, model)
        similar_users = get_top_k(similarity[user_inx], k, user_inx)
        predictions = []
        for game_inx in range(num_games):
            predictions.append(predict(data, similarity, similar_users, user_inx, game_inx))
        # print(model)
        print(predictions)
        print(get_top_k(predictions, 10, user_inx))
        print(rmse(data[user_inx, :].A[0], predictions))


def load_data(file_name):
    return sparse.load_npz(FILE_PATH + file_name)


def learn_all(data, metric):
    # user by game
    num_users, num_games = np.shape(data)
    sim = np.zeros((num_users, num_users))

    for u1 in range(num_users):
        sim[u1] = learn_row(u1, data, metric)
        # if u1 % 10 == 0:
            # print(u1)

    return sim


def learn_row(row_inx, A, metric):
    length, width = np.shape(A)
    data_1 = A[row_inx, :].A[0]
    arr = []
    for i in range(length):
        data_2 = A[i, :].A[0]
        if np.count_nonzero(data_1) != 0 and np.count_nonzero(data_2) != 0:
            if metric == 'cosine':
                arr.append(1 - scipy.spatial.distance.cosine(data_1, data_2))
            elif metric == 'jaccard':
                arr.append(1 - scipy.spatial.distance.jaccard(data_1, data_2))
            else:
                r, p = scipy.stats.pearsonr(data_1, data_2)
                if math.isnan(r):
                    r = 0
                arr.append(r)
        else:
            arr.append(0)

    return arr


# returns indices of top k closest users in ascending order except i (same user)
def get_top_k(A, k, i):
    temp = np.copy(A)
    temp[i] = -2  # all other elements are [-1, 1]
    ind = np.argpartition(temp, -k)[-k:]
    return ind[np.argsort(temp[ind])[::-1]]


# indices of similar users
def predict(data, similarity, similar_users, user_inx, game_inx):
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


def rmse(actual, predicted):
    return mean_squared_error(actual, predicted)


# run
main()
