from surprise import SVD, Reader, Dataset, KNNBasic, KNNWithMeans, NormalPredictor, NMF, accuracy
from surprise.model_selection import cross_validate
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
                r = 0
                if i[1] >= 1251:
                    r = 5
                elif i[1] >= 361:
                    r = 4
                elif i[1] >= 107:
                    r = 3
                elif i[1] >= 27:
                    r = 2
                else:
                    r = 1
                f.write('%s\t%d\t%d\n' % (user, i[0], r))
    f.close()

    with open('data/test_users.p', 'rb') as f:
        users = pickle.load(f)

    f = open('data/test_users.csv', 'w')

    for user in users:
        if users[user] != {}:
            for i in users[user].items():
                r = 0
                if i[1] >= 1251:
                    r = 5
                elif i[1] >= 361:
                    r = 4
                elif i[1] >= 107:
                    r = 3
                elif i[1] >= 27:
                    r = 2
                else:
                    r = 1
                f.write('%s\t%d\t%d\n' % (user, i[0], r))
    f.close()

    reader = Reader(line_format='user item rating', sep='\t', rating_scale=(1, 5))

    train_data = Dataset.load_from_file('data/train_users.csv', reader=reader) \
                  .build_full_trainset()

    # reader = Reader(line_format='user item rating', sep='\t', rating_scale=(1, 5))

    # test_data = Dataset.load_from_file('data/test_users.csv', reader=reader) \
    #               .build_full_trainset()

    with open('data/test_users.csv', 'r') as f:
        s = list(map(lambda x: tuple(x.split('\t')), f.read().split('\n')))
        test_data = []
        for x in s:
            if len(x) > 2:
                test_data.append((x[0], x[1], float(x[2])))


    #print(data.ur)

    #algo = KNNBasic(sim_options={'name': 'cosine'})
    # algo = NMF(verbose=True)
    algo = SVD(verbose=True)
    # algo = NormalPredictor()
    algo.fit(train_data)

    # #cross_validate(algo, data, verbose=True)

    # print(algo.predict('76561197960675902', '70', r_ui=63, verbose=True))
    # print(algo.predict('76561197960675902', '4540', r_ui=22, verbose=True))
    # print(algo.predict('76561197960675902', '550', r_ui=791, verbose=True))
    # print(algo.predict('76561197960675902', '10190', r_ui=1253, verbose=True))
    # print(algo.predict('76561197960675902', '10', r_ui=1037, verbose=True))

    predictions = algo.test(test_data)

    print(accuracy.rmse(predictions))

main()