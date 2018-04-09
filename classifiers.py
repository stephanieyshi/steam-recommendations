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

def main():
    with open('user_ids.txt', 'r') as f:
        l = list(set(map(int, f.read().split())))
        # extract number of friends each
        # number of achievements for a game
        # ^ get games per person by GetOwnedGames

    user_games = {}
    for i in l:
        print i
        new_url = BASE_URL + '/IPlayerService/GetOwnedGames/v0001/?key=' + key + '&steamid=' + str(i) + '&format=json'
        r = requests.get(new_url)
        # filter out games that they bought but haven't played?

        data = r.json()['response']
        try:
            games = [j for j in data['games'] if j['playtime_forever'] != 0]
            # user_vector = [0] * 3000
            # for j in games:
            #     pass
            user_games[i] = games
        except:
            continue

    with open('user_games.json', 'w') as f:
        json.dump(user_games, f, separators = (',', ':'))

    print len(user_games)

main()

def classify_hours(X, user, k, game_id):
    X = np.array(X)
    dist = np.sum(np.square(X - user), 2)
    knn = np.argpartition(dist, k)

    neighbor_hours = X[knn, game_id - 1]
    return np.sum(neighbor_hours)[0] / len(neighbor_hours)
