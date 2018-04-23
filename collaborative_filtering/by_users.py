#!/usr/bin/env python3
import numpy as np
import math
import random
import scipy.stats
import scipy.sparse as sparse
from statistics import mean
from sklearn.metrics import mean_squared_error

# variables
FILE_PATH = ""
TEST_DATA = FILE_PATH + "test_user_mat.npz"


def main():
    ks = [5, 10, 15, 25, 50]
    test_data = load_data(TEST_DATA)
    num_users, num_games = np.shape(test_data)

    # pick some random users
    users = []
    for i in range(5):
        users.append(random.randint(0, num_users - 1))

    models = ['pearson', 'cosine', 'jaccard']

    for model in models:
        print('TRAINING MODEL: ' + model)
        total_rmse = [0, 0, 0, 0, 0]
        for user_inx in users:
            print('PREDICTING FOR USER ' + str(user_inx))
            similarity = learn_row(user_inx, test_data, model)
            for k in ks:
                print('VALUE OF K: ' + str(k))
                similar_users = get_top_k(similarity, k, user_inx)
                predictions_test = []
                for game_inx in range(num_games):
                    predictions_test.append(predict(test_data, similarity, similar_users, user_inx, game_inx))
                this_rmse = rmse(test_data[user_inx, :].A[0], predictions_test)
                print('TEST ERROR FOR USER ' + str(user_inx) + ': ' + str(this_rmse))
                if k == 5:
                    total_rmse[0] += this_rmse
                elif k == 10:
                    total_rmse[1] += this_rmse
                elif k == 15:
                    total_rmse[2] += this_rmse
                elif k == 25:
                    total_rmse[3] += this_rmse
                else:
                    total_rmse[4] += this_rmse

        print('TOTAL SUM OF ERRORS:')
        print(total_rmse)
        avg_rmse = [math.sqrt(x/5) for x in total_rmse]
        print('RMSE:')
        print(avg_rmse)


def load_data(file_name):
    return sparse.load_npz(FILE_PATH + file_name)


def learn_all(data, metric):
    # user by game
    num_users, num_games = np.shape(data)
    sim = sparse.zeros((num_users, num_users))

    for u1 in range(num_users):
        sim[u1] = learn_row(u1, data, metric)

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
        sum += similarity[other_user] * (data[other_user, game_inx] - m)

    # want to predict
    denom = np.sum(np.absolute(similarity)[similar_users])

    return mu + sum / denom


def mean_prediction(data, user_inx):
    return mean(data[user_inx, :].A[0])


def rmse(actual, predicted):
    return mean_squared_error(actual, predicted)


# run
main()
