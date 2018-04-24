#!/usr/bin/env python3
import numpy as np
import math
import random
import scipy.sparse as sparse
from statistics import mean

# variables
FILE_PATH = ""
TEST_DATA = FILE_PATH + "test_user_mat.npz"


def main():
    ks = [5, 10, 15, 25, 50]
    test_data = load_data(TEST_DATA)
    num_users, num_games = np.shape(test_data)
    num_test_users = 100

    # pick some random users
    users = []
    for i in range(num_test_users):
        users.append(random.randint(0, num_users - 1))

    models = ['pearson', 'cosine']

    for model in models:
        print('TRAINING MODEL: ' + model)
        total_rmse = [0, 0, 0, 0, 0]
        for user_inx in users:
            print('PREDICTING FOR USER ' + str(user_inx))
            for k in ks:
                actual_test = []
                predictions_test = []
                for game_inx in range(num_games):
                    similarity = learn_by_game(game_inx, test_data, model)
                    if test_data[user_inx, game_inx] != 0:
                        # try to predict this based on the similar items
                        actual_test.append(test_data[user_inx, game_inx])
                        similar_games = get_top_k_game(test_data, similarity, k, user_inx, game_inx)
                        predictions_test.append(predict(test_data, similarity, similar_games, user_inx, game_inx))
                this_rmse = math.sqrt(se(actual_test, predictions_test) / len(actual_test))
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
        avg_rmse = [(x / num_test_users) for x in total_rmse]
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
                sim = calculate_cosine(data_1, data_2)
                arr.append(sim)
            else:
                sim = calculate_pearson(data_1, data_2)
                arr.append(sim)
        else:
            arr.append(0)

    return arr


def learn_by_game(col_inx, A, metric):
    length, width = np.shape(A)
    data_1 = A[:, col_inx].A[0]
    arr = []
    for i in range(length):
        data_2 = A[:, col_inx].A[0]
        if np.count_nonzero(data_1) != 0 and np.count_nonzero(data_2) != 0:
            if metric == 'cosine':
                sim = calculate_cosine(data_1, data_2)
                arr.append(sim)
            else:
                sim = calculate_pearson(data_1, data_2)
                arr.append(sim)
        else:
            arr.append(0)

    return arr


def calculate_pearson(u1, u2):
    mu_1 = mean(u1)
    mu_2 = mean(u2)

    # intersection
    intersection = []
    for i in range(len(u1)):
        if u1[i] != 0 and u2[i] != 0:
            intersection.append(i)

    num = 0
    denom_u1 = 0
    denom_u2 = 0

    for game in intersection:
        num += (u1[game] - mu_1) * (u2[game] - mu_2)
        denom_u1 += (u1[game] - mu_1) ** 2
        denom_u2 += (u2[game] - mu_2) ** 2

    denom = math.sqrt(denom_u1 * denom_u2)
    if denom == 0:
        return 0
    else:
        return num / denom


def calculate_cosine(u1, u2):
    # intersection
    intersection = []
    for i in range(len(u1)):
        if u1[i] != 0 and u2[i] != 0:
            intersection.append(i)

    num = 0
    denom_u1 = 0
    denom_u2 = 0

    for game in intersection:
        num += u1[game] * u2[game]
        denom_u1 += u1[game] ** 2
        denom_u2 += u2[game] ** 2

    denom = math.sqrt(denom_u1 * denom_u2)
    if denom == 0:
        return 0
    else:
        return num / denom


def get_top_k_game(user_game_data, similarities, k, user_inx, game_inx):
    similarities = np.array(similarities)
    # print(similarities)
    data_csr = user_game_data.tocsr()
    indices = np.array((data_csr[:, game_inx] != 0).todense().nonzero()[0])
    actual_similarities = similarities[indices]
    x = actual_similarities.size
    if x >= k:
        ind = np.argpartition(actual_similarities, -k)[-k:]
    else:
        ind = np.argpartition(actual_similarities, -x)[-x:]
    return indices[ind[np.argsort(actual_similarities[ind])[::-1]]]


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


def se(actual, predicted):
    return np.sum((np.array(actual) - np.array(predicted)) ** 2)


# run
main()
