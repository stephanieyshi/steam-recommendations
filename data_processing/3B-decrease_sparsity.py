import pickle
import numpy as np
import scipy.sparse as sparse
import math

# Parameters
# Path to git repo on your machine
directory_path = "C:/Users/bpiv4/Dropbox/CIS520/cis520/"
target_name = "08"  # Name of density to be generated
type_name = "test"  # train or test
target = .08

mat_path = directory_path + 'data/' + type_name + '_user_mat' + '.npz'
games_path = directory_path + 'data/' + type_name + '_games.p'
users_map_path = directory_path + 'data/' + type_name + '_user_map.p'
output_mat_path = directory_path + 'data/' + \
    type_name + '_user_mat_' + target_name + '.npz'
output_games_path = directory_path + "data/" + \
    type_name + '_games_' + target_name + ".p"
output_users_map_path = directory_path + "data/" + \
    type_name + '_user_map_' + target_name + ".p"
input_users_path = directory_path + 'data/' + type_name + '_users.p'
output_users_path = directory_path + "data/" + \
    type_name + '_users_' + target_name + ".p"


users_mat = sparse.load_npz(mat_path)
entries = users_mat.data
print(len(entries))

users_mat = users_mat.todense()
print("Max hours: " + str(users_mat.max()))

games = {}
with open(games_path, 'rb') as f:
    games = pickle.load(f)
    f.close()

users_map = {}
with open(users_map_path, 'rb') as f:
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
sparsity = total / (inverse_sparsity.shape[1] * n)
print(sparsity)
remove_cols = []
org_cols = inverse_sparsity.shape[1]
print("Done column arrangement")

# Reverse user dictionary

# Rearrange matrix to be in order of most dense to least dense user rows
user_sparsity = np.count_nonzero(users_mat, 1)
# print(user_sparsity)
ordered_users = np.argsort(user_sparsity, 0)[::-1]
# for new_index in ordered_users:
#   print(new_index)

# print(ordered_users.shape)
print("Done user ordering")
# print(users_mat.shape)
users_mat = np.squeeze(users_mat[ordered_users.flatten(), :])
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
# for entry in user_nonzero:
#   print(entry)
user_remove_rate = math.ceil(n / org_cols)
user_batches = 0

print("Beginning sparsity computation")

while sparsity < target:
    print(sparsity)
    # Remove game with the lowest sparsity
    targ_col = ordered_cols[0, 0]
    # print(targ_col)
    remove_cols.append(targ_col)
    total = total - inverse_sparsity[0, targ_col]
    # print(total)
    # print(type(total))

    # Remove k users with lowest sparsity
    user_batches = user_batches + 1
    ceiling_entries_removed = user_nonzero[((user_nonzero.shape[0]-1) -
                                            (user_remove_rate*user_batches)):
                                           (user_nonzero.shape[0]-1) -
                                           user_remove_rate*(user_batches-1)]
    ceiling_entries_removed = np.sum(
        ceiling_entries_removed) - user_remove_rate
    ceiling_entries_removed = max(ceiling_entries_removed, 0)
    total = total - ceiling_entries_removed
    sparsity = total/((inverse_sparsity.shape[1] - len(remove_cols)) *
                      (n-(user_batches*user_remove_rate)))
    # print(sparsity)
    # print(user_batches)
    # print(user_remove_rate)
    # print(n-(user_batches*user_remove_rate))
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

# Remove relevant users from the matrix:
users_mat = users_mat[0:users_mat.shape[0]-user_batches*user_remove_rate, ]

print("Done updating matrix")

# Some sanity checks
new_cols = users_mat.shape[1]
print(sparsity)
print(len(remove_cols))
print(org_cols - new_cols)
print(n - users_mat.shape[0])
print(user_batches * user_remove_rate)

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
sparse.save_npz(output_mat_path, output)
print(output.shape)

# Delete old entries in user_map
# THIS IS A PROBLEM
new_size = users_mat.shape[0]
for user in list(users_map.keys()):
    new_index = users_map[user]
    if new_index >= new_size:
        del users_map[user]

# update games index map
pickle_out = open(output_games_path, 'wb')
pickle.dump(games, pickle_out)
pickle_out.close()

# update users index map
pickle_out = open(output_users_map_path, 'wb')
pickle.dump(users_map, pickle_out)
pickle_out.close()

# open users map
users = {}
with open(input_users_path, 'rb') as f:
    users = pickle.load(f)
    f.close()

final_users = {}
for user, game_dict in users.items():
    if user in users_map:
        final_users[user] = {}
        for game, hours in game_dict.items():
            if game in games:
                final_users[user][game] = hours


# for user, game_dict in final_users.items():
#     for game, hours in game_dict.items():
#         print(type(hours))

# update users index map
pickle_out = open(output_users_path, 'wb')
pickle.dump(final_users, pickle_out)
pickle_out.close()
