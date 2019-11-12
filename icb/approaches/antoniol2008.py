from skfeature.function.information_theoretical_based.FCBF import fcbf

from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from icb.utils import split_camel_case, porter_stemming


class AntoniolCountVectorizer(CountVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(porter_stemming(split_camel_case(tokenize(doc))))


class SymmetricalUnvertaintyAttributeSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.selected_features = []

    def fit(self, X, y, **fit_params):
        self.selected_features, _ = fcbf(X.toarray(), y)
        return self

    def transform(self, X, y=None, **fit_params):
        return X[:, self.selected_features]


class Antoniol2008(BaseEstimator, ClassifierMixin):
    """
    Reimplementation of approach by Antoniol et al. (2008): "Is it a bug or an enhancement?: a text-based approach to classify change requests." CASCON.
    DOI: https://doi.org/10.1145/1463788.1463819
    """

    def __init__(self, clf):
        """
        :param clf: Original paper used Naive bayes, Logistic regression, and Alternating decision trees
        """
        self.clf = clf

    def fit(self, X, y):
        self.text_clf = Pipeline([
            ('vect', AntoniolCountVectorizer(stop_words=None)),
            ('feat_select', SymmetricalUnvertaintyAttributeSelector()),
            ('clf', self.clf)
        ])
        self.text_clf.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.text_clf.predict(X)

    def filter(self, df):
        return df['description']