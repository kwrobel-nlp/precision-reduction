import sys
import random
import numpy as np
import time
import csv
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.base import TransformerMixin

import load

SVD_COMPONENTS = [100, 200, 300, 400, 500, 1000]
PRECISIONS = [16, 8, 7, 6, 5, 4, 3, 2, 1]
CLASS_WEIGHT = 'balanced' # None or balanced

def reduce_precision(X, r):
    m = 2 ** (r - 1)

    result = X * m
    result = np.ceil(result)
    result /= m
    return result


class Reductor(TransformerMixin):
    def __init__(self, r):
        self.r = r

    def transform(self, X, *_):
        return reduce_precision(X, r=self.r)

    def fit(self, *_):
        return self


class Data:
    def __init__(self, X_train, X_test, y_train, y_test, target_names=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target_names = target_names
        self.model_name = ['', '', '']
        self.time = {}

    def mname(self):
        return '_'.join(self.model_name)


class M1_TFIDF:
    @staticmethod
    def apply(data, max_features):  # tfidf
        start_time = time.time()
        count_vectorizer = CountVectorizer(max_features=max_features)

        X_train_counts = count_vectorizer.fit_transform(data.X_train)
        print('Words', len(count_vectorizer.vocabulary_))
        tfidf_transformer = TfidfTransformer().fit(X_train_counts)
        X_train_tfidf = tfidf_transformer.transform(X_train_counts)
        data.time['1_train'] = time.time() - start_time

        start_time = time.time()
        X_test_counts = count_vectorizer.transform(data.X_test)
        X_test_tfidf = tfidf_transformer.transform(X_test_counts)
        data.time['1_test'] = time.time() - start_time

        print('M1_TFIDF Time - train: %f test: %f' % (data.time['1_train'], data.time['1_test']))

        data.time['test'] = 0.0
        data.time['train'] = 0.0

        data.X_train_tfidf = X_train_tfidf
        data.X_test_tfidf = X_test_tfidf

        data.X_train_out = X_train_tfidf
        data.X_test_out = X_test_tfidf

        data.model_name[0] = 'TFIDF'
        data.model_name[1] = ''
        data.model_name[2] = ''

class M2_TFIDF_SVD:
    @staticmethod
    def apply(data, components):  # tfidf + svd
        start_time = time.time()
        svd = TruncatedSVD(components)
        lsa = make_pipeline(svd, Normalizer(copy=False))

        X_train_svd = lsa.fit_transform(data.X_train_tfidf)
        data.time['2_train'] = time.time() - start_time

        start_time = time.time()
        X_test_svd = lsa.transform(data.X_test_tfidf)
        data.time['2_test'] = time.time() - start_time

        data.time['test'] = data.time['2_test']
        data.time['train'] = data.time['2_train']

        data.X_train_svd = X_train_svd
        data.X_test_svd = X_test_svd

        data.X_train_out = X_train_svd
        data.X_test_out = X_test_svd

        data.model_name[1] = 'SVD(%d)' % components
        data.model_name[2] = ''

class M3_TFIDF_SVD_RED:
    @staticmethod
    def apply(data, r):  # tfidf + svd + reduction
        start_time = time.time()
        X_train_reduced = Reductor(r=r).transform(data.X_train_svd)
        data.time['3_train'] = time.time() - start_time

        start_time = time.time()
        X_test_reduced = Reductor(r=r).transform(data.X_test_svd)
        data.time['3_test'] = time.time() - start_time

        data.time['test'] = data.time['2_test'] + data.time['3_test']
        data.time['train'] = data.time['2_train'] + data.time['3_train']

        data.X_train_out = X_train_reduced
        data.X_test_out = X_test_reduced

        data.model_name[2] = 'R(%d)' % r


class M4_TFIDF_RED:
    @staticmethod
    def apply(data, r):  # tfidf + reduction
        start_time = time.time()
        X_train_reduced = Reductor(r=r).transform(data.X_train_tfidf)
        data.time['4_train'] = time.time() - start_time

        start_time = time.time()
        X_test_reduced = Reductor(r=r).transform(data.X_test_tfidf)
        data.time['4_test'] = time.time() - start_time

        data.time['test'] = data.time['4_test']
        data.time['train'] = data.time['4_train']

        data.X_train_reduced = X_train_reduced
        data.X_test_reduced = X_test_reduced

        data.X_train_out = X_train_reduced
        data.X_test_out = X_test_reduced

        data.model_name[1] = 'R(%d)' % r
        data.model_name[2] = ''


class M5_TFIDF_RED_SVD:
    @staticmethod
    def apply(data, components):  # tfidf + reduction + svd
        start_time = time.time()
        svd = TruncatedSVD(components)
        lsa = make_pipeline(svd, Normalizer(copy=False))

        X_train_svd = lsa.fit_transform(data.X_train_reduced)
        data.time['5_train'] = time.time() - start_time

        start_time = time.time()
        X_test_svd = lsa.transform(data.X_test_reduced)
        data.time['5_test'] = time.time() - start_time

        data.time['test'] = data.time['4_test'] + data.time['5_test']
        data.time['train'] = data.time['4_train'] + data.time['5_train']

        data.X_train_out = X_train_svd
        data.X_test_out = X_test_svd

        data.model_name[2] = 'SVD(%d)' % components





corpus_names = sys.argv[1]
names = [corpus_names.split(' ')]

f = open('results_' + corpus_names+'.csv', 'w')
writer = csv.writer(f)

for corpus_name in names:
    X, y, ind = load.load(corpus_name)
    print(corpus_name, len(X))

for corpus_name in names:
    random.seed(42)

    X, y, ind = load.load(corpus_name)
    target_names = sorted(ind.index.keys())
    X, y = shuffle(X, y, random_state=42)
    length = len(X)

    for i, (train, test) in enumerate(KFold(length, n_folds=5)):
        X_train = np.take(X, train)
        y_train = np.take(y, train)

        X_test = np.take(X, test)
        y_test = np.take(y, test)

        data = Data(X_train, X_test, y_train, y_test)

        models = [(M1_TFIDF, None)]
        for x in SVD_COMPONENTS:
            models.append((M2_TFIDF_SVD, x))
            for x in PRECISIONS:
                models.append((M3_TFIDF_SVD_RED, x))

        for x in PRECISIONS:
            models.append((M4_TFIDF_RED, x))
            for x in SVD_COMPONENTS:
                models.append((M5_TFIDF_RED_SVD, x))

        for model, arg in models:
            try:
                model.apply(data, arg)
            except ValueError:
                continue

            classifiers = [(KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='cosine'), 'KNN1'),
                           (KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine'), 'KNN5'),
                           (LinearSVC(loss='hinge', class_weight=CLASS_WEIGHT), 'SVM'),
                           (LogisticRegression(class_weight=CLASS_WEIGHT), 'LR')]
            for classifier, clf_name in classifiers:
                start_time = time.time()
                classifier.fit(data.X_train_out, data.y_train)
                data.time['6_train'] = time.time() - start_time

                start_time = time.time()
                predicted = classifier.predict(data.X_test_out)
                data.time['6_test'] = time.time() - start_time

                micro = precision_score(data.y_test, predicted, average='micro')
                macro = precision_score(data.y_test, predicted, average='macro')

                f1 = f1_score(data.y_test, predicted, average='macro')
                f1w = f1_score(data.y_test, predicted, average='weighted')
                rmacro = recall_score(data.y_test, predicted, average='macro')

                print(
                    '%s Model: %s, Arg: %s, Classifier: %s, Fold: %d, PrecisionMicro: %f, PrecisionMacro: %f, RecallMacro: %f, F1: %f, F1w: %f, Train: %f + %f, Test: %f + %f' % (
                        corpus_name, data.mname(), arg, clf_name, i, micro, macro, rmacro, f1, f1w, data.time['train'],
                        data.time['6_train'], data.time['test'], data.time['6_test']))
                writer.writerow(
                    [corpus_name, data.mname(), clf_name, i, micro, macro, rmacro, f1, f1w, data.time['train'],
                     data.time['6_train'], data.time['test'], data.time['6_test']])
                f.flush()
                #print(metrics.classification_report(data.y_test, predicted, target_names=target_names)) #confusion matrix

        data = None