from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import collections


class Indexer:
    def __init__(self):
        self.index = {}
        self.i = 0

    def get_index(self, label):
        if label not in self.index:
            self.index[label] = self.i
            self.i += 1

        return self.index[label]


def load(name):
    train_file = 'datasets/' + name + '-train-stemmed.txt'
    test_file = 'datasets/' + name + '-test-stemmed.txt'

    i = Indexer()
    X = []
    y = []
    for path in [train_file, test_file]:
        f = open(path, 'rt')
        for line in f:
            row = line.split('\t')
            label = row[0]
            text = row[1]
            idd = i.get_index(label)
            X.append(text)
            y.append(idd)
    return X, y, i


if __name__ == "__main__":
    names = ['20ng', 'r8', 'r52', 'webkb', 'cade']

    for name in names:
        X, y, ind = load(name)
        lenghts = [len(x) for x in X]
        print(name)
        print('Documents', len(X))
        print('Categories', len(ind.index))
        count_vectorizer = CountVectorizer()
        X_train_counts = count_vectorizer.fit_transform(X)
        print('Vocabulary', len(count_vectorizer.vocabulary_))
        print('Avg document length', np.mean(lenghts))

        cls = collections.defaultdict(int)
        for yy in y:
            cls[yy] += 1

        print(sorted(cls.items(), key=lambda x: x[1]))

# 20ng
# Documents 18821
# Categories 20
# Vocabulary 70213
# Avg document length 851.699378354

# r8
# Documents 7674
# Categories 8
# Vocabulary 17387
# Avg document length 390.566849101

# r52
# Documents 9100
# Categories 52
# Vocabulary 19241
# Avg document length 418.107582418

# webkb
# Documents 4199
# Categories 4
# Vocabulary 7770
# Avg document length 909.568706835

# cade
# Documents 40983
# Categories 12
# Vocabulary 193997
# Avg document length 913.002196033