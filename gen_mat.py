import pickle
import numpy as np
import scipy.sparse as sparse

games = []
with open('./data/games.p', 'rb') as f: 
  games = pickle.load(f)
  f.close()

users = {}
with open('./data/users.p', 'rb') as f: 
  users = pickle.load(f)
  f.close()
users_list = list(users.keys())
num_users = len(users)
num_games = len(games)

print(str(num_users))
print(str(num_games))

user_mat = sparse.lil_matrix((num_users, num_games))
i = 0
for user, games_dict in users.items():
  games_ar = np.zeros((1, len(games_dict))) 
  hours_ar = np.zeros((1, len(games_dict)))
  j = 0
  for game, hours in games_dict.items(): 
    games_ar[0, j] = games[game]
    hours_ar[0, j] = hours
    j = j + 1
  user_mat[i, games_ar] = hours_ar
  if i % 10000 == 0: 
    print(str(i))
  i = i + 1

user_mat = sparse.csc_matrix(user_mat)
sparse.save_npz('./data/user_mat.npz', user_mat)