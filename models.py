from surprise import SVD, Reader, Dataset, KNNBasic, KNNWithMeans, NormalPredictor
from collections import defaultdict
import pickle
import os


def main():
    # with open('data/train_users.p', 'rb') as f:
    #     users = pickle.load(f)

    # f = open('data/train_users.csv', 'w')

    # for user in users:
    #     if users[user] != {}:
    #         for i in users[user].items():
    #             f.write('%s\t%d\t%d\n' % (user, i[0], i[1]))
    # f.close()

    reader = Reader(line_format='user item rating', sep='\t', rating_scale=(0, 2091943))

    data = Dataset.load_from_file('data/train_users.csv', reader=reader) \
                  .build_full_trainset()

    #algo = KNNBasic(sim_options={'name': 'cosine'})
    algo = SVD()
    algo.fit(data)

    print(algo.predict('76561197960675902', '70', r_ui=63, verbose=True))
    print(algo.predict('76561197960675902', '4540', r_ui=22, verbose=True))
    print(algo.predict('76561197960675902', '550', r_ui=791, verbose=True))
    print(algo.predict('76561197960675902', '10190', r_ui=1253, verbose=True))
    print(algo.predict('76561197960675902', '10', r_ui=1037, verbose=True))

main()