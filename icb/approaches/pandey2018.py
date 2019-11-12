from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from icb.utils import DenseTransformer, StemmingStopWordRemovalCountTokenizer


class Pandey2018(BaseEstimator, ClassifierMixin):
    """
    Reimplementation of approach by Pandey et al. (2018): "Automated Classification of Issue Reports from a Software Issue Tracker". In: Progress in Intelligent Computing Techniques: Theory, Practice, and Applications.
    DOI: https://doi.org/10.1007/978-981-10-3373-5_42
    """

    def __init__(self, clf):
        """
        :param clf: Original paper used Naive Bayes, SVM, Logistic regression, linear discriminant analysis
        """
        self.clf = clf

    def fit(self, X, y):
        self.text_clf = Pipeline([
            # Exclude numbers (see Figure 1 in Paper)
            ('vect', StemmingStopWordRemovalCountTokenizer(token_pattern=r'\b[^\d\W]+\b')),
            # There should be some threshold in here that excludes terms with a frequency lower than it,
            # but it is not stated what this threshold is
            ('to_dense', DenseTransformer()),
            ('clf', self.clf)
        ])
        self.text_clf.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.text_clf.predict(X)

    def filter(self, df):
        return df['title']
