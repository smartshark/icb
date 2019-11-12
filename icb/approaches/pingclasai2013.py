import numpy as np

from icb.utils import clean_html, remove_punctuation, StemmingStopWordRemovalCountTokenizer

from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation


class LDADataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        maximum_values = np.argmax(X, axis=1)
        X.fill(0)
        i = 0

        for highest_topic_prop in maximum_values:
            X[i][highest_topic_prop] = 1
            i += 1
        return X


class Pingclasai2013(BaseEstimator, ClassifierMixin):
    """
    Reimplementation of approach by Pingclasai et al. (2013): "Classifying Bug Reports to Bugs and Other Requests Using Topic Modeling". In: Proceedings of the 20th Asia-Pacific Software Engineering Conference (APSEC).
    DOI: https://doi.org/10.1109/APSEC.2013.105
    """

    # Paper determined optimal number of topics to be 50
    NUM_TOPICS = 50

    def __init__(self, clf):
        """
        :param clf: Original paper used alternating decision trees, naive bayes and logistic regression
        """
        self.clf = clf

    def fit(self, X, y=None):
        self.text_clf = Pipeline([
            ('vect', StemmingStopWordRemovalCountTokenizer()),
            ('lda', LatentDirichletAllocation(n_components=self.NUM_TOPICS)),
            ('transform_lda_data', LDADataTransformer()),
            ('clf', self.clf)
        ])
        self.text_clf.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.text_clf.predict(X)

    def filter(self, df):
        df['complete'] = df['title']+" "+df['description']+" "+df['discussion']
        df['complete'] = df['complete'].map(lambda x: clean_html(x))
        df['complete'] = df['complete'].map(lambda x: remove_punctuation(x))
        return df.complete
