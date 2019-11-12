from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from icb.utils import porter_stemming, StemmingStopWordRemovalCountTokenizer, DenseTransformer

# Extracted from the original paper
FEATURE_SET = [
    'enhancement',
    'refactoring',
    'improvement',
    'performance',
    'feature',
    'usability',
    'efficiency',
    'contribution',
    'reduction',
    'support',
    'update',
    'modification',
    'add',
    'change',
    'configuration'
]


class Otoom2019(BaseEstimator, ClassifierMixin):
    """
    Reimplementation of approach by Otoom et al. (2019): "Automated Classification of Software Bug Reports." In: Proceedings of the 9th International Conference on Information Communication and Management.
    DOI: https://doi.org/10.1145/3357419.3357424
    """

    def __init__(self,  clf):
        """
        :param clf: Original paper used Naive Bayes, SVM, Random Forest
        """
        self.clf = clf

    def fit(self, X, y):
        self.text_clf = Pipeline([
            # In the original paper they do not say that they stem the feature set.
            # However it would be useless otherwise
            ('vect', StemmingStopWordRemovalCountTokenizer(vocabulary=porter_stemming(FEATURE_SET))),
            ('to_dense', DenseTransformer()),
            ('clf', self.clf)
        ])
        self.text_clf.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.text_clf.predict(X)

    def filter(self, df):
        return df['title'] + " " + df['description']
