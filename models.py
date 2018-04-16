from surprise import SVD, Reader, Dataset
from collections import defaultdict
import pickle
import os


def main():
    with open('data/train_users.p', 'rb') as f:
        users = pickle.load(f)

    f = open('data/train_users.csv', 'w')

    for user in users:
        if users[user] != {}:
            for i in users[user].items():
                f.write('%s\t%d\t%d\n' % (user, i[0], i[1]))
    f.close()



    #file_path = os.path.expanduser('data/train_users.csv')

    reader = Reader(line_format='user item rating', sep='\t')

    data = Dataset.load_from_file('data/train_users.csv', reader=reader) \
                  .build_full_trainset()

    algo = SVD()

    # modified_users = defaultdict(list, {x : list(i.items()) for x, i in users.items()})

    algo.fit(data)
    #print(data.ur)

main()