import json
from pprint import pprint 
import pickle

data = json.load(open('./data/temp/user_games.json'))
print(type(data))
print(len(data))

print(type(data[list(data.keys())[0]]))
pprint(data[list(data.keys())[0]][0])

all_games = []
for user in data.keys():
  game_dict = {}
  game_tuples = data[user]
  for game in game_tuples:
    game_dict[game[0]] = game[1]
    if game[0] not in all_games:
      all_games.append(game[0])
  data[user] = game_dict

# Pickle the user dictionary 
pickle_out = open("./data/temp/users1.pickle", 'wb')
pickle.dump(data, pickle_out)
pickle_out.close() 

#Pickle the list of games
games_pickle = open("./data/temp/games1.pickle", "wb")
pickle.dump(all_games, games_pickle)
games_pickle.close()
