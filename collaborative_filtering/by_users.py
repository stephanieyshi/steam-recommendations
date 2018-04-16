#!/usr/bin/env python3
import pickle
import numpy
import scipy.stats

# variables
FILE_PATH = "./data/"


def main():
    dev_users, train_users, test_users = load_data()
    learn(dev_users)
    print('DONE')


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
    R = numpy.zeros(n, n) # n x n matrix of zeros

    for key, value in get_items(users):
        continue

    # TODO: calculate mean rating for each user

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
