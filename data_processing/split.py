import pickle
import random
import scipy
import re

# Parameter
directory_path = "C:/Users/bpiv4/Dropbox/CIS520/cis520/"
size_test = 20000

# open users map
users = {}
with open(directory_path + 'data/train_users.csv', 'rb') as f:
    for line in f:
        line = line.strip()
        line = line.rstrip('\t')
        line = re.split(line, r'\t+')
        user = line[0].strip()
        game = line[1].strip()
        hours = line[2].strip()
        if users[user] is None:
            game_dict = {}
            game_dict[game] = hours
            users[user] = game_dict
        else:
            game_dict = users[user]
            game_dict[game] = hours
            users[user] = game_dict
    f.close()

with open(directory_path + 'data/test_users.csv', 'rb') as f:
    for line in f:
        line = line.strip()
        line = line.rstrip('\t')
        line = re.split(line, r'\t+')
        user = line[0].strip()
        game = line[1].strip()
        hours = line[2].strip()
        if users[user] is None:
            game_dict = {}
            game_dict[game] = hours
            users[user] = game_dict
        else:
            game_dict = users[user]
            game_dict[game] = hours
            users[user] = game_dict
    f.close()

train_users = {}
train_games = {}

test_users = {}
test_games = {}

test_user_list = random.sample(list(users.keys()), size_test)

for user in test_user_list:
    test_users[user] = users[user]
    del users[user]

train_users = users

game_count = 0
for users, game_dict in train_users.items():
    for game, hours in game_dict.items():
        train_games[game] = game_count
        game_count = game_count + 1

game_count = 0
for users, game_dict in test_users.items():
    for game, hours in game_dict.items():
        test_games[game] = game_count
        game_count = game_count + 1

pickle_out = open(directory_path + "data/train_games.p", 'wb')
pickle.dump(train_games, pickle_out)
pickle_out.close()

pickle_out = open(directory_path + "data/test_games.p", 'wb')
pickle.dump(test_games, pickle_out)
pickle_out.close()

pickle_out = open(directory_path + "data/train_users.p", 'wb')
pickle.dump(train_users, pickle_out)
pickle_out.close()

pickle_out = open(directory_path + "data/test_users.p", 'wb')
pickle.dump(test_users, pickle_out)
pickle_out.close()
