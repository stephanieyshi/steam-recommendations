import numpy as np 
import pickle 
import math
import seaborn as sns
import scipy.sparse as sparse
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
directory_path = "C:/Users/bpiv4/Dropbox/CIS520/cis520/"

with open(directory_path + 'data/games.p', 'rb') as f: 
  games = pickle.load(f)
  f.close()

users_map = {}
with open(directory_path + 'data/user_map.p', 'rb') as f: 
  users_map = pickle.load(f)
  f.close()

users = {}
with open(directory_path + 'data/users.p', 'rb') as f: 
  users = pickle.load(f)
  f.close()

users_mat = sparse.load_npz(directory_path + 'data/user_mat.npz')
entries = users_mat.data

org_max = np.amax(entries)
indices = np.argwhere(entries == org_max)
print(indices.size)
entries = np.delete(entries, indices)
print(np.amax(entries))

new_users = {}
new_games = {}
j = 0
count = 0
emp_count = 0
for user, game_dict in users.items():
  curr_dict ={}
  if len(game_dict) == 0: 
    emp_count = emp_count + 1
  for game, hours in list(game_dict.items()):
    if hours < 200000:
      curr_dict[game] = hours
      if game not in new_games:
        new_games[game] = j
        j = j + 1
    else: 
      count = count + 1
  if len(curr_dict) > 0:
    new_users[user] = curr_dict

print(len(new_users))
print(len(new_games))
print("Count: " + str(count))
print("Empty user count: " + str(emp_count))

# while curr_max == org_max:
#   i = i + 1
#   entries = np.delete(entries, curr_max)
#   curr_max = np.amax(entries)
# print(str(i))
#update users index map

pickle_out = open(directory_path + "data/clean_users.p", 'wb')
pickle.dump(new_users, pickle_out)
pickle_out.close()

pickle_out = open(directory_path + "data/clean_games.p", 'wb')
pickle.dump(new_games, pickle_out)
pickle_out.close()
