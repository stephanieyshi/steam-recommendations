import pickle
import random
import scipy
import re
import math

# Parameter
directory_path = "C:/Users/bpiv4/Dropbox/CIS520/cis520/"
size_test = 20000

# open users map
users = {}
total = 0

with open(directory_path + 'data/train_users.csv', 'rb') as f:
    for line in f:
        line = line.decode('utf-8')
        line = line.strip()
        line = re.split('\t+', line)
        user = line[0].strip()
        game = line[1].strip()
        hours = line[2].strip()
        total = total + 1
        if user not in users:
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
        line = line.decode('utf-8')
        line = line.strip()
        line = re.split('\t+', line)
        user = line[0].strip()
        game = line[1].strip()
        hours = line[2].strip()
        total = total + 1
        if user not in users:
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

test_game_count = 0
train_game_count = 0
for users, game_dict in users.items():
    for game, hours in game_dict.items():
        decision = random.random()
        if decision < .1:
            if game not in test_games:
                test_games[game] = test_game_count
                test_game_count = test_game_count + 1
            if user not in test_users:
                test_users[user] = {}
            test_users[user][game] = hours
        else:
            if game not in train_games:
                train_games[game] = train_game_count
                train_game_count = train_game_count + 1
            if user not in train_users:
                train_users[user] = {}
            train_users[user][game] = hours

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
