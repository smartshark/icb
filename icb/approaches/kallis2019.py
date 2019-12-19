import numpy as np

from skift import ColLblBasedFtClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from icb.utils import remove_new_lines


class Kallis2019(BaseEstimator, ClassifierMixin):
    """
    Reimplementation of approach by Kallis et al. (2019): "Ticket Tagger: Machine Learning Driven Issue Classification." In: Proceedings of the 35th IEEE International Conference on Software Maintenance and Evolution
    DOI: (not yet assigned)
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        self.sk_clf = ColLblBasedFtClassifier(input_col_lbl='combined', wordNgrams=1, minCount=14)
        self.sk_clf.fit(X, y)
        return self

    def predict(self, X, y=None):
        return np.around(self.sk_clf.predict_proba(X)[:, 1], decimals=0)

    def filter(self, df):
        df['combined'] = df['title'] + " " + df['description']
        df['combined'] = df['combined'].map(lambda x: remove_new_lines(x))
        return df[['combined']]