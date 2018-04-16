import pickle
import numpy as np
import scipy.sparse as sparse
import math

#Parameters
target = .05  # Gives the target sparsity
directory_path = "C:/Users/bpiv4/Dropbox/CIS520/cis520/" # Path to git repo on your machine

users_mat = sparse.load_npz(directory_path + 'data/user_mat.npz').todense()
games = []
with open(directory_path + 'data/games.p', 'rb') as f: 
  games = pickle.load(f)
  f.close()

users_map = {}
with open(directory_path + 'data/user_map.p', 'rb') as f: 
  users_map = pickle.load(f)
  f.close()

n = users_mat.shape[0]
# print(type(n))
inverse_sparsity = np.count_nonzero(users_mat, 0)
# print(inverse_sparsity.shape)
ordered_cols = np.argsort(inverse_sparsity)
total = np.sum(inverse_sparsity)
# print(type(total))
# print(total)
sparsity = total/(inverse_sparsity.shape[1] * n)
remove_cols = []
org_cols = inverse_sparsity.shape[1]
print("Done column arrangement")

#Reverse user dictionary

#Rearrange matrix to be in order of most dense to least dense user rows
user_sparsity = np.count_nonzero(users_mat, 1)
# print(user_sparsity)
ordered_users = np.argsort(user_sparsity, 0)[::-1]
# for new_index in ordered_users:
#   print(new_index)

# print(ordered_users.shape)
print("Done user ordering")
# print(users_mat.shape)
users_mat = np.squeeze(users_mat[ordered_users.flatten(),:])
print("Done user matrix rearrangement")
# print(users_mat.shape)
# Updating user_map
reveresed_user_map = {}
for user, index in users_map.items():
  # print(user)
  # print(index)
  reveresed_user_map[index] = user

# print(len(reveresed_user_map))
# print(type(reveresed_user_map))

users_map = {}
for new_index in range(ordered_users.shape[0]):
  old_index = ordered_users[new_index, 0]
  # print(old_index)
  # print(new_index)
  user_id = reveresed_user_map[old_index]
  users_map[user_id] = new_index

print("Done user map update")
print(len(users_map))
print(type(users_map))

user_nonzero = np.count_nonzero(users_mat, 1)
user_remove_rate = math.ceil(n / org_cols)
user_batches = 0

print("Beginning sparsity computation")

while sparsity < target: 
  # Remove game with the lowest sparsity
  targ_col = ordered_cols[0, 0]
  # print(targ_col)
  remove_cols.append(targ_col)
  total = total - inverse_sparsity[0, targ_col]
  # print(total)
  # print(type(total))

  # Remove k users with lowest sparsity
  user_batches = user_batches + 1
  ceiling_entries_removed = user_nonzero[((user_nonzero.shape[0]-1) - user_remove_rate*user_batches):
    (user_nonzero.shape[0]-1) - user_remove_rate*(user_batches-1)]
  ceiling_entries_removed = np.sum(ceiling_entries_removed) - user_remove_rate
  ceiling_entries_removed = max(ceiling_entries_removed, 0)

  sparsity = total/((inverse_sparsity.shape[1] - len(remove_cols)) *
   (n-(user_batches*user_remove_rate)))
  ordered_cols = np.delete(ordered_cols, 0, 1)
  # print(ordered_cols.shape[1])

print("Done sparsity computation")

# Update game-column indexes 
for game in list(games.keys()):
  game_ind = games[game]
  if game_ind in remove_cols:
    del games[game]
  else:
    counter = 0
    for col in remove_cols:
      if game_ind > col:
        counter = counter + 1
    games[game] = games[game] - counter

print("Done games map update")

# Remove relevant columns from matrix: 
users_mat = np.delete(users_mat, np.array(remove_cols), 1)

#Remove relevant users from the matrix:
users_mat = users_mat[0:users_mat.shape[0]-user_batches*user_remove_rate,]

print("Done updating matrix")

# Some sanity checks
new_cols = users_mat.shape[1]
print(sparsity)
print(len(remove_cols))
print(org_cols - new_cols)
print(n - users_mat.shape[0])
print(user_batches*user_remove_rate)

curr_users = users_mat.shape[0]

col_num = users_mat.shape[1]
problem = False
for game, index in games.items():
  if index >= col_num:
    problem = True
if not problem:
  print('Everything is okay')
else:
  print("There's an indexing failure")
output = sparse.csc_matrix(users_mat)
sparse.save_npz(directory_path + 'data/dense_user_mat.npz', output)

# Delete old entries in user_map 
new_size = users_mat.shape[0]
for user in list(users_map.keys()): 
  new_index = users_map[user]
  if new_index >= new_size:
    del users_map[user]

#update games index map
pickle_out = open(directory_path + "data/games_dense.p", 'wb')
pickle.dump(games, pickle_out)
pickle_out.close()

#update users index map
pickle_out = open(directory_path + "data/user_map_dense.p", 'wb')
pickle.dump(users_map, pickle_out)
pickle_out.close()