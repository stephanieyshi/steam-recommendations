import numpy as np
import pickle
import scipy.sparse as sparse
import math

# Parameters
# Path to git repo on your machine
directory_path = "C:/Users/bpiv4/Dropbox/CIS520/cis520/"
target_name = "08"  # Name of density to be generated
type_name = "train"  # train or test
target = .08

users_map_path = directory_path + 'data/' + \
    type_name + '_users_' + target_name + '.p'
with open(users_map_path, 'rb') as f:
    users = pickle.load(f)
    f.close()
for user, game_dict in users.items():
    for game, hours in game_dict.items():
        game_dict[game] = float(hours)

pickle_out = open(users_map_path, 'wb')
pickle.dump(users, pickle_out)
pickle_out.close()
