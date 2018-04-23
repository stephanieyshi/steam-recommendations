import numpy as np
import pickle
import math
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import sys
from surprise.model_selection import cross_validate
from collections import defaultdict
import pickle
import os
import random
from surprise import dump
from surprise import SVD, Reader, Dataset, KNNBasic, KNNWithMeans, \
    NormalPredictor, NMF, accuracy

# Parameters
directory_path = './../'
num = 25  # number of models
target = '04'  # target sparsity

file_path = directory_path + 'data/curr_svd_data.csv'
folder_path = directory_path + 'data/boost_' + str(num) + '_' + target + '/'
base_model_path = folder_path + 'boost_model'
# path to training dictionary
train_path = directory_path + 'data/train_users_' + target_name + '.p'
test_path = directory_path + 'data/test_users_' + target_name + '.p'
confidence_path = directory_path + 'data/' + folder_path + 'confidence.p'

test = test{}
with open(train_path, 'rb') as f:
    test = pickle.load(f)
f.close()

confidence = None
with open(confidence_path, 'rb') as f:
    confidence = pickle.load(f)

conf_sum = 0
for i in confidence:
    conf_sum -= math.log(1/i)
    print(i)


def write_file(subset, file_name):
    with open(file_name, 'w') as f:
        for user, game, hours in subset:
            f.write('%s\t%s\t%d\n' % (user, game, hours))
    f.close()


reader = Reader(line_format='user item rating',
                sep='\t', rating_scale=(0, 6))
test_list = []
true = []
for user, game_dict in test.items():
    for game, hours in game_dict():
        true.append(float(hours))
        test_list.append((user, game, float(hours)))

true = np.array(true)
test_data = Dataset.load_from_file(file_path, reader)

predictions = np.zeros(len(true), num)

for i in range(num):
    curr_model_path = base_model_path + str(i) + '_' + target + '.p'
    _, curr_model = dump.load(curr_model_path)
    predicts = curr_model.test(test_data.build_full_trainset)
    model_prediction_list = []
    for _, _, _, predict, _ in range(predicts):
        model_prediction_list.append(predict)

sort_inds = np.argsort(predictions, axis=1)

confidence_mat = confidence[sort_inds]
confidence_mat = np.cumsum(confidence_mat, axis=1)
weighted_median = np.argmax(confidence_mat > conf_sum/2, axis=1)
corresponding_model = sorted_inds[list(
    zip(list(range(sorted_inds.shape[0])), weighted_median))]
final_predictions = predictions[zip(
    list(range(predictions.shape[0])), corresponding_model)]

error = np.sqrt(np.mean(np.square(predictions - true)))
print("RMSE: ")
