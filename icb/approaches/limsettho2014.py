from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.pipeline import Pipeline

from icb.utils import StemmingStopWordRemovalCountTokenizer


class Limsettho2014(BaseEstimator, ClassifierMixin):
    """
    Reimplementation of approach by Limsettho et al. (2014): "Comparing hierarchical dirichlet process with latent dirichlet allocation in bug report multiclass classification". In: Proceedings of the 15th IEEE/ACIS International Conference on Software Engineering, Artificial Intelligence, Networking and Parallel/Distributed Computing (SNPD)
    DOI: https://doi.org/10.1109/SNPD.2014.6888695

    We only implemented the LDA approach, as the paper states that it is superior to the HDP approach (in case of performance)
    """

    def __init__(self, clf, num_topics=100):
        """
        :param clf: Original paper used alternating decision trees, naive bayes, and logistic regression
        """
        self.clf = clf
        self.num_topics = num_topics

    def fit(self, X, y):
        self.text_clf = Pipeline([
            ('vect', StemmingStopWordRemovalCountTokenizer()),
            ('lda', LatentDirichletAllocation(n_components=self.num_topics)),
            ('clf', self.clf)
        ])
        self.text_clf.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.text_clf.predict(X)

    def filter(self, df):
        return df['title'] + " " + df['description'] + " " + df['discussion']
