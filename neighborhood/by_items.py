#!/usr/bin/env python3
import numpy as np
import math
import random
import scipy.sparse as sparse
from statistics import mean

# variables
FILE_PATH = ""
TEST_DATA = FILE_PATH + "test_user_mat_02.npz"


def main():
    ks = [5, 10, 15, 25, 50]
    test_data = load_data(TEST_DATA)
    num_users, num_games = np.shape(test_data)
    num_test_users = 1

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
            actual_test = []
            predictions_test_5 = []
            predictions_test_10 = []
            predictions_test_15 = []
            predictions_test_25 = []
            predictions_test_50 = []
            for game_inx in range(num_games):
                if test_data[user_inx, game_inx] != 0:
                    print('GAME ' + str(game_inx))
                    print(test_data[user_inx, game_inx])
                    actual_test.append(test_data[user_inx, game_inx])
                    similarity = learn_by_game(game_inx, test_data, model)
                    # print(similarity)
                    for k in ks:
                        similar_games = get_top_k_game(test_data, similarity, k, user_inx, game_inx)
                        prediction = predict(test_data, similarity, similar_games, user_inx, game_inx)
                        if k == 5:
                            predictions_test_5.append(prediction)
                        elif k == 10:
                            predictions_test_10.append(prediction)
                        elif k == 15:
                            predictions_test_15.append(prediction)
                        elif k == 25:
                            predictions_test_25.append(prediction)
                        else:
                            predictions_test_50.append(prediction)

            total_rmse[0] += math.sqrt(se(actual_test, predictions_test_5) / len(actual_test))
            total_rmse[1] += math.sqrt(se(actual_test, predictions_test_10) / len(actual_test))
            total_rmse[2] += math.sqrt(se(actual_test, predictions_test_15) / len(actual_test))
            total_rmse[3] += math.sqrt(se(actual_test, predictions_test_25) / len(actual_test))
            total_rmse[4] += math.sqrt(se(actual_test, predictions_test_50) / len(actual_test))

        avg_rmse = [(x / num_test_users) for x in total_rmse]
        print('RMSE:')
        print(avg_rmse)


def load_data(file_name):
    return sparse.load_npz(FILE_PATH + file_name)


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


# return the similarities of a game to all other games
def learn_by_game(col_inx, A, metric):
    length, width = np.shape(A)
    data_1 = A[:, col_inx].transpose().toarray()[0]
    arr = []
    for i in range(width):
        data_2 = A[:, i].transpose().toarray()[0]
        # print(data_2)
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


def calculate_pearson(g1, g2):
    mu_1 = mean(g1)
    mu_2 = mean(g2)

    # intersection
    intersection = []
    for i in range(len(g1)):
        if g1[i] != 0 and g2[i] != 0:
            intersection.append(i)

    num = 0
    denom_g1 = 0
    denom_g2 = 0

    for user in intersection:
        num += (g1[user] - mu_1) * (g2[user] - mu_2)
        denom_g1 += (g1[user] - mu_1) ** 2
        denom_g2 += (g2[user] - mu_2) ** 2

    denom = math.sqrt(denom_g1 * denom_g2)
    if denom == 0:
        return 0
    else:
        return num / denom


def calculate_cosine(g1, g2):
    # intersection
    intersection = []
    for i in range(len(g1)):
        if g1[i] != 0 and g2[i] != 0:
            intersection.append(i)

    num = 0
    denom_g1 = 0
    denom_g2 = 0

    for user in intersection:
        num += g1[user] * g2[user]
        denom_g1 += g1[user] ** 2
        denom_g2 += g2[user] ** 2

    denom = math.sqrt(denom_g1 * denom_g2)
    if denom == 0:
        return 0
    else:
        return num / denom


def get_top_k_game(user_game_data, similarities, k, user_inx, game_inx):
    temp = np.copy(similarities)
    temp[game_inx] = -2
    ind = np.argpartition(temp, -k)[-k:]
    return ind[np.argsort(temp[ind])[::-1]]


# indices of similar games
def predict(data, similarity, similar_games, user_inx, game_inx):
    sum = 0
    for other_game in similar_games:
        # compute mean of prediction of this user
        sum += similarity[other_game] * data[user_inx, other_game]

    # want to predict
    denom = np.sum(np.absolute(similarity)[similar_games])

    return sum / denom


def mean_prediction(data, user_inx):
    return mean(data[user_inx, :].A[0])


def se(actual, predicted):
    return np.sum((np.array(actual) - np.array(predicted)) ** 2)


# run
main()
