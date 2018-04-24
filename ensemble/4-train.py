import numpy as np
import pickle
import math
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import random

# Parameters
# path to directory
# Parameters
# path to directory
directory_path = "./../"
data_directory = directory_path + 'data/'
k = 10

with open(data_directory + 'train_bias_08.p', 'rb') as f:
    biases = pickle.load(f)
mu = biases[0]
buvec = biases[1]
bivec = biases[2]

init_mu = biases[0]
init_buvec = biases[1]
init_bivec = biases[2]

del biases

with open(data_directory + 'qi.p', 'rb') as f:
    q = pickle.load(f)
f.close()

# path to directory
with open(data_directory + 'pu.p', 'rb') as f:
    p = pickle.load(f)
f.close()

# path to directory
with open(data_directory + 'train_shrunk_pearson_08.p', 'rb') as f:
    pearson = pickle.load(f)
    pearson = pearson + pearson.transpose()
f.close()

user_mat = sparse.load_npz(data_directory + 'train_user_mat_08.npz')
coo_user_mat = user_mat.tocoo()
# name of file containing global mean, user bias, and item bias
type_name = "train"  # train or test
size_name = "04"  # name of data density
shrinkage = 100  # Beta in shrinkage formula
k = 100
gam = 0.007
gam2 = 0.07
gam3 = 0.007
lam6 = 0.05
lam7 = 0.015
lam8 = 0.015
Rk = 10  # This is obviously made up placeholder
step = 0
wij = np.zeros((712, 712)) + 0.01
count = 0
num_epochs = 15
epoch_size = 50000

for epoch in range(num_epochs):
    epoch_subset = random.sample(zip(coo_user_mat.row, coo_user_mat.col,
                                     coo_user_mat.data), epoch_size)
    for user, item, rui in epoch_subset:
        rel_pearson = pearson[item, :]
        order = np.argsort(rel_pearson)
        orderdescending = order[::-1]
        uservect = user_mat[user, :]
        bu = buvec[0, user]
        bi = bivec[0, item]
        bui = mu + bu + bi
        uhb = uservect[0, orderdescending[1:k]] != 0
        Rk = np.sum(uhb)
        Rk = Rk + 1 * (Rk == 0)
        Rk_term = (1/(np.sqrt(Rk)))
        inner_term = uservect[0, orderdescending[1:k+1]] - \
            (init_mu+init_buvec[0, user_ind] +
             init_bivec[0, orderdescending[1:k+1]])
        inner_sum = uhb.multiply(uservect[0, orderdescending[1:k]] -
                                 (init_mu+init_bu[0, user] +
                                  init_bivec[0, orderdescending[1:k]]))
        summation = np.dot(inner_sum.toarray(),
                           wij[orderdescending[1:k+1], game_ind])
        rhat = bui + np.dot(q[item, :], p[user, :]) + Rk_term * summation

        eui = rui - rhat

        buvec[0, user] = bu + gam * (eui - lam6*bu)
        bivec[0, item] = bi + gam * (eui - lam6*bi)
        q[item, :] = q[item, :] + \
            (gam2 * ((eui * p[user, :]) - lam7 * q[item, :]))
        p[user, :] = p[user, :] + \
            (gam2 * ((eui * q[item, :]) - lam7 * p[user, :]))
        wij[item, orderdescending[1:k+1]] = wij[item,
                                                orderdescending[1:k+1]] + \
            gam3 * (Rk_term * eui * (
                uservect[0, orderdescending[1:k+1]] - (
                    mu+bu+bivec[0, orderdescending[1:k+1]])) -
                    lam8*wij[item, orderdescending[1:k+1]])

        if (count % 1000 == 0):
            print(count)
            print(epoch)
        count = count + 1

output = {}
output['bu'] = buvec
output['bi'] = bivec
output['pearson'] = pearson
output['mu'] = mu
output['p'] = p
output['q'] = q
output['wij'] = wij

pickle_out = open(directory_path + 'data/ensemble_dump_08.p', 'wb')
pickle.dump(output, pickle_out)
pickle_out.close()
