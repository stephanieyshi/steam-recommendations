#!/usr/bin/python
import json
import numpy as np

# variables
FILE_PATH = "./data/"

def load_data():
    with open('user_games.json', 'r') as f:
        games = json.loads(f.read())
    X, users = [], []
    for user in games.keys():
        sparse_vector = [0] * 70000
        for i in games[user]:
            if i['appid'] < 700000:
                sparse_vector[i['appid'] / 10] = i['playtime_forever']
        X.append(sparse_vector)
        users.append(user)
    print X, users

def learn():
    print('done')

load_data()
learn()
