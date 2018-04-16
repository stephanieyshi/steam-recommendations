#!/usr/bin/env python3
import pickle

# variables
FILE_PATH = "./data/"


def load_data():
    with open(FILE_PATH + 'train_users.p', 'rb') as f:
        s = pickle.load(f)
    print('LOADING DATA...')

    # TODO: load data
    print('DATA LOADED')


def learn():
    print('hello')
    # TODO: calculate mean rating for each user

    # TODO: calculate Pearson correlation coefficient

    # TODO: calculate neighborhood-based predictor function


def evaluate():
    print('hello')
    # TODO: calculate MAE
    # TODO: calculate RMSE


load_data()
print('DONE')
