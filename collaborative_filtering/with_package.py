#!/usr/bin/env python3
import math
import scipy.sparse as sparse
import scipy.stats
from sklearn.neighbors import NearestNeighbors
# variables
FILE_PATH = "../data/"


def main():
    data = load_data('user_mat.npz')
    k = 5
    find_nearest_neighbors(data, k, 'correlation')
    print('DONE')


def load_data(file_name):
    print('LOADING DATA')
    return sparse.load_npz(FILE_PATH + file_name)


def learn_row(row1, row2):
    # return value indicating distance between them\
    r, p = scipy.stats.pearsonr(row1.toarray(), row2.toarray())
    if math.isnan(r):
        return 0
    return r


def find_nearest_neighbors(data, k, metric):
    # num_users, num_games = np.shape(data)
    neigh = NearestNeighbors(n_neighbors=k, metric=learn_row)
    neigh.fit(data)
    # this doesn't really work


main()
