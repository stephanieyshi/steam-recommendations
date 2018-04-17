import pickle
import random
import scipy

# Combine games data into a single games list
games1 = []
games2 = []
with open('./data/temp/games1.pickle', 'rb') as f:
    games1 = pickle.load(f)
    f.close()

with open('./data/temp/games2.pickle', 'rb') as r:
    games2 = pickle.load(r)
    r.close
games = list(set(games1 + games2))
final_games = {}
i = 0
for game in games:
    final_games[game] = i
    i = i + 1

# Combine users data into a single users list
with open('./data/temp/users1.pickle', 'rb') as f:
    users1 = pickle.load(f)
    f.close()

with open('./data/temp/users2.pickle', 'rb') as f:
    users2 = pickle.load(f)
    f.close()

print(len(users2))
print(len(users1))
print(type(users1[list(users1.keys())[0]]))
print(len(set(list(users1.keys()) + list(users2.keys()))))
users = {**users1, **users2}
del users1, users2
print(len(users))

pickle_out = open("./data/users.p", 'wb')
pickle.dump(users, pickle_out)
pickle_out.close()

pickle_out = open("./data/games.p", 'wb')
pickle.dump(final_games, pickle_out)
pickle_out.close()

print("All done pickling")

# Split sample of users into train, test, and dev
users_list = list(users.keys())
test_users = random.sample(users_list, 200000)
dev_users = test_users[100000:]
test_users = test_users[0:100000]

test_data = {}
dev_data = {}

print("Data split")

for user in test_users:
    test_data[user] = users[user]
    del users[user]
for user in dev_users:
    dev_data[user] = users[user]
    del users[user]

print("Dictionary split")

pickle_out = open("./data/train_users.p", 'wb')
pickle.dump(users, pickle_out)
pickle_out.close()

pickle_out = open("./data/dev_users.p", 'wb')
pickle.dump(dev_data, pickle_out)
pickle_out.close()

pickle_out = open("./data/test_users.p", 'wb')
pickle.dump(test_data, pickle_out)
pickle_out.close()
