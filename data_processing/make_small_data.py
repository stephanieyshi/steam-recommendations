import pickle
import numpy as np
import scipy.sparse as sparse
import math

# Parameters
directory_path = "C:/Users/bpiv4/Dropbox/CIS520/cis520/" # Path to git repo on your machine
n = 500 # Number of users

users_map = {}
with open(directory_path + 'data/train_users.p', 'rb') as f: 
  users_map = pickle.load(f)
  f.close()

users_map = dict(list(users_map.items())[:n])

all_games = {}
game_counter = 0
for user, game_dict in users_map.items():
  for game in game_dict.keys():
    if game not in all_games:
      all_games[game] = game_counter
      game_counter = game_counter + 1

pickle_out = open(directory_path + 'data/small_user_map.p', 'wb')
pickle.dump(users_map, pickle_out)
pickle_out.close()

pickle_out = open(directory_path + 'data/small_games.p', 'wb')
pickle.dump(all_games, pickle_out)
pickle_out.close()
