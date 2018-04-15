import pickle
import numpy as np
import scipy.sparse as sparse

#Parameters
target = .05  # Gives the target sparsity

users_mat = sparse.load_npz('./data/user_mat.npz').todense()
games = []
with open('./data/games.p', 'rb') as f: 
  games = pickle.load(f)
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

while sparsity < target: 
  targ_col = ordered_cols[0, 0]
  # print(targ_col)
  remove_cols.append(targ_col)
  total = total - inverse_sparsity[0, targ_col]
  # print(total)
  # print(type(total))
  sparsity = total/((inverse_sparsity.shape[1] - len(remove_cols)) * n)
  ordered_cols = np.delete(ordered_cols, 0, 1)
  # print(ordered_cols.shape[1])

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

# Remove relevant columns from matrix: 
users_mat = np.delete(users_mat, np.array(remove_cols), 1)

# Some sanity checks
# new_cols = users_mat.shape[1]
# print(sparsity)
# print(len(remove_cols))
# print(org_cols - new_cols)
# print(str(users_mat.shape))

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
sparse.save_npz('dense_user_mat.npz', output)

#update games index map
pickle_out = open("./data/games_dense.p", 'wb')
pickle.dump(games, pickle_out)
pickle_out.close()