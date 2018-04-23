from surprise.model_selection import cross_validate
from collections import defaultdict
import pickle
import os
import numpy as np
from scipy import stats
from surprise import SVD, Reader, Dataset, KNNBasic, \
    KNNWithMeans, NormalPredictor, NMF, accuracy


def main():
    # Load full training data
    with open('data/train_users_04.p', 'rb') as f:
        users = pickle.load(f)

    f = open('data/train_users.csv', 'w')

    for user in users:
        if users[user] != {}:
            # zscores = stats.zscore(list(users[user].values()))
            # items = [(a[0], zscores[i]) for i, a
            #  in enumerate(users[user].items())]
            for i in users[user].items():
                t = float(i[1])
                if t > 6:
                    t = 6
                elif t < 1:
                    t = 1
                f.write('%s\t%s\t%.03f\n' % (user, i[0], t))
    f.close()

    print("finished train data")

    # Load full test data
    with open('data/test_users_04.p', 'rb') as f:
        users = pickle.load(f)

    f = open('data/test_users.csv', 'w')

    for user in users:
        if users[user] != {}:
            # zscores = stats.zscore(list(users[user].values()))
            # items = [(a[0], zscores[i]) for i,
            #  a in enumerate(users[user].items())]
            for i in users[user].items():
                t = float(i[1])
                if t > 6:
                    t = 6
                elif t < 1:
                    t = 1
                f.write('%s\t%s\t%.03f\n' % (user, i[0], t))
    f.close()

    print("finished test data")

    ######### START OF TRAINING #########

    reader = Reader(line_format='user item rating',
                    sep='\t', rating_scale=(1, 6))

    train_data = Dataset.load_from_file('data/train_users.csv', reader=reader)
    #                  .build_full_trainset()

    with open('data/test_users.csv', 'r') as f:
        s = list(map(lambda x: tuple(x.split('\t')), f.read().split('\n')))
        test_data = []
        for x in s:
            if len(x) > 2:
                test_data.append((x[0], x[1], float(x[2])))

    # print(data.ur)

    # algo = KNNBasic(sim_options={'name': 'cosine'})
    # algo = NMF(verbose=True)
    algo = SVD(verbose=True)
    # algo = NormalPredictor()
    algo.fit(train_data.build_full_trainset())

    # cross_validate(algo, train_data, verbose=True)

    # print(algo.predict('76561197960675902', '70', r_ui=63, verbose=True))
    # print(algo.predict('76561197960675902', '4540', r_ui=22, verbose=True))
    # print(algo.predict('76561197960675902', '550', r_ui=791, verbose=True))
    # print(algo.predict('76561197960675902', '10190', r_ui=1253, verbose=True))
    # print(algo.predict('76561197960675902', '10', r_ui=1037, verbose=True))

    predictions = algo.test(test_data)

    print(accuracy.rmse(predictions))

    # with open('pu.p', 'wb') as f:
    #     pickle.dump(algo.pu, f)
    # with open('qi.p', 'wb') as f:
    #     pickle.dump(algo.qi, f)

    return

    # g = open('results2.txt','w')

    # for num in range(3,5):
    #     with open('data/train_users_0%d.p' % num, 'rb') as f:
    #         users = pickle.load(f)
    #     f = open('data/train_users_0%d.csv' % num, 'w')
    #     for user in users:
    #         if users[user] != {}:
    #             for i in users[user].items():
    #                 r = 0
    #                 if i[1] >= 1251:
    #                     r = 5
    #                 elif i[1] >= 361:
    #                     r = 4
    #                 elif i[1] >= 107:
    #                     r = 3
    #                 elif i[1] >= 27:
    #                     r = 2
    #                 else:
    #                     r = 1
    #                 f.write('%s\t%d\t%d\n' % (user, i[0], r))
    #     f.close()
    #     reader = Reader(line_format='user item rating', sep='\t', rating_scale=(1, 5))

    #     train_data = Dataset.load_from_file('data/train_users_0%d.csv' % num, reader=reader) \
    #               .build_full_trainset()

    #     with open('data/test_users.csv', 'r') as f:
    #         s = list(map(lambda x: tuple(x.split('\t')), f.read().split('\n')))
    #         test_data = []
    #         for x in s:
    #             if len(x) > 2:
    #                 test_data.append((x[0], x[1], float(x[2])))
    #     algo = SVD(verbose=True)

    #     algo.fit(train_data)

    #     predictions = algo.test(test_data)

    #     print('%d:' % num)
    #     print(accuracy.rmse(predictions))
    #     g.write('%d:' % num)
    #     g.write(str(accuracy.rmse(predictions)))
    #     g.write('\n')
    # g.close()


main()
