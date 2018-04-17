#!/usr/bin/env python3
import pickle
import numpy as np
import scipy.stats

# variables
FILE_PATH = "./data/"


def main():
    dev_users, train_users, test_users = load_data()
    small_users = load_small_data()
    learn(small_users)
    print('DONE')


def load_small_data():
    print('LOADING SMALL DATA')
    with open(FILE_PATH + 'small_user_map.p', 'rb') as f:
        small_users = pickle.load(f)
    print('SMALL DATA LOADED')
    return small_users


def load_data():
    print('LOADING DATA...')
    with open(FILE_PATH + 'dev_users.p', 'rb') as f:
        dev_users = pickle.load(f)
    with open(FILE_PATH + 'train_users.p', 'rb') as f:
        train_users = pickle.load(f)
    with open(FILE_PATH + 'test_users.p', 'rb') as f:
        test_users = pickle.load(f)
    print('DATA LOADED')
    return dev_users, train_users, test_users


def learn(users):
    print('LEARNING...')
    # initialize matrix
    n = len(users)
    print(n)

    sim_pearson = np.zeros((n, n))
    mean_rating = []
    user_map = []

    for key

    # calculate mean rating for each user
    # we use hours played as a proxy for the mean rating for each user
    for key, value in get_items(users):
        user_map.append(key)

        hours = 0
        for k, v in get_items(value):
            hours += v

        if len(value) != 0:
            mean = hours / len(value)
        else:
            mean = 0

        mean_rating.append(mean)

    print(mean_rating)

    # TODO: calculate Pearson correlation coefficient
    # use scipy.stats.pearsonr(x, y)

    # TODO: calculate neighborhood-based predictor function
    print('MODEL LEARNED')


def predict():
    print('PREDICTING')


def evaluate():
    print('hello')
    # TODO: calculate MAE
    # TODO: calculate RMSE


def get_items(dict_object):
    for key in dict_object:
        yield key, dict_object[key]


# run
main()
