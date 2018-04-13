from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import requests
import json
import numpy as np

key = 'F17E5E95FBA4B361BCFE2278F87CFDE0'

BASE_URL = 'http://api.steampowered.com'

def scrape_ids(id):
    new_url = BASE_URL + '/ISteamUser/GetFriendList/v0001/?key=' + key + '&steamid=' + str(id) + '&relationship=friend'
    r = requests.get(new_url)
    return map(lambda x: int(x['steamid']), r.json()['friendslist']['friends'])

def scrape_features():
    with open('user_ids.txt', 'r') as f:
        l = list(set(map(int, f.read().split())))

    user_games = {}
    for i in l:
        print i
        new_url = BASE_URL + '/IPlayerService/GetOwnedGames/v0001/?key=' + key + '&steamid=' + str(i) + '&format=json'
        r = requests.get(new_url)

        data = r.json()['response']
        try:
            games = [j for j in data['games'] if j['playtime_forever'] != 0]
            user_vector = [0] * 3000
            user_games[i] = games
        except:
            continue

    with open('user_games.json', 'w') as f:
        json.dump(user_games, f, separators = (',', ':'))

    print len(user_games)

#main()

def main():
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

def classify_hours(X, user, k, game_id):
    X = np.array(X)
    dist = np.sum(np.square(X - user), 2)
    knn = np.argpartition(dist, k)
    neighbor_hours = X[knn, game_id - 1]
    return np.sum(neighbor_hours)[0] / len(neighbor_hours)

print main()