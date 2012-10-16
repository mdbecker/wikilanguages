# -*- coding: utf8 -*-

import cPickle
import bz2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import numpy as np


def get_label_encoder(labels):
    """Generate a label encoder given a list of labels.

    This can be used to convert text labels to numerical values.

    """
    le = LabelEncoder()
    le.fit(labels)
    return le


def encode_targets(y1, y2, le):
    """Convert text labels to numerical values using a LabelEncoder."""

    y1_t = le.transform(y1)
    y2_t = le.transform(y2)

    return y1_t, y2_t


def load_data():
    """Load the wikipedia articles."""

    f = bz2.BZ2File('features.pkl.bz2', 'rb')
    data = cPickle.load(f)
    f.close()
    return data['X1'], data['X2'], data['y1'], data['y2']


def svc(X1, X2, y1_t, y2_t):
    """Generate a Support Vector Machine classifier with a linear kernal."""
    text_clf = Pipeline([
        (
            'vect',
            TfidfVectorizer(
                analyzer='char',
                binary=True,
                lowercase=False,
                max_features=50000,
                ngram_range=(2, 3),
                norm='l2',
                stop_words=None,
                use_idf=True,
            ),
        ),
        ('clf', LinearSVC(loss='l2', C=1, dual=True, verbose=2)),
    ])
    model = text_clf.fit(X1, y1_t)

    print 'fit done'

    predicted = text_clf.predict(X2)
    print np.mean(predicted == y2_t)

    return text_clf


def sgd(X1, X2, y1_t, y2_t):
    """Generate a Stochastic Gradient Descent classifier."""
    text_clf = Pipeline([
        (
            'vect',
            TfidfVectorizer(
                analyzer='char',
                binary=True,
                lowercase=False,
                max_features=50000,
                ngram_range=(2, 3),
                norm='l2',
                stop_words=None,
                use_idf=True,
            ),
        ),
        (
            'clf',
            SGDClassifier(
                alpha=0.0000061,
                fit_intercept=False,
                loss='hinge',
                n_iter=67,
                penalty='l1',
                shuffle=True,
                warm_start=True,
            ),
        ),
    ])
    model = text_clf.fit(X1, y1_t)

    print 'fit done'

    predicted = text_clf.predict(X2)
    print np.mean(predicted == y2_t)

    return text_clf


def generate_label_encoder(labels):
    """Given a bunch of labels, generate a LabelEncoder for them."""
    labels = set(labels)
    le = get_label_encoder(labels)
    return le

if __name__ == "__main__":
    """Example usage"""
    X1, X2, y1, y2 = load_data()
    le = generate_label_encoder(list(y1) + list(y2))
    y1_t, y2_t = encode_targets(y1, y2, le)
    svc_clf = svc(X1, X2, y1_t, y2_t)
    sgd_clf = sgd(X1, X2, y1_t, y2_t)
    print sgd_clf.predict([
        u'Insert some text to classify here!',
        u'Insérer du texte à classer ici!',
        u'Infoga en text att klassificera här!',
        u'Inserire un testo di classificare qui!',
        u'Legen Sie einen Text hier zu klassifizieren!',
        u'Introduzca un texto para clasificar aquí!',
        u'Introduceți un text pentru a clasifica aici!',
    ])
    # ['en', 'fr', 'sv', 'it', 'de', 'es', 'ro']
