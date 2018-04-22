import numpy as np
import pickle
import math
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# Parameters
directory_path = "/Users/Chris/Downloads/train_user_mat_08.npz"  # path to directory

user_mat = sparse.load_npz(directory_path)
coo_user_mat = user_mat.tocoo()
# name of file containing global mean, user bias, and item bias
type_name = "train"  # train or test
size_name = "04"  # name of data density
shrinkage = 100  # Beta in shrinkage formula

k = 100
gam = 0.007
gam2 = 0.007
gam3 = 0.007
lam6 = 0.005 
lam7 = 0.015

for user, item, rui in zip(coo_user_mat.row, coo_user_mat.col, coo_user_mat.data):
    r = user_mat[user,item]
    eui = r - rhat
    bu = bu + gam * (eui - lam6*bu)
    bi = bi + gam * (eui - lam6*bi)
    qi = qi + gam2 * ((eui * pu) - lam7 * qi)
    pu = pu + gam2 * (eui * qi - lam7 * pu)
    wij = wij + gam3 * ((1/(Rk^(1/2)))* eui * (r - buj) - lam8*wij)
    rhat = mu + bu + bi + qi*pu + (1/(Rk^(1/2))) * 1


