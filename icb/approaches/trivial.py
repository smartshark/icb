import numpy as np

from sklearn.base import BaseEstimator


class Trivial(BaseEstimator):
    """
    Trivial estimator that predicts everything as bug.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.ones(shape=len(X))

    def filter(self, df):
        return df
