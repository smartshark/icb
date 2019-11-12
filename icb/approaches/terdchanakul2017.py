import os
import subprocess
import tempfile

from skfeature.function.statistical_based.CFS import cfs
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from icb.utils import remove_programming_characters


class CorrelationBasedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.selected_features = []

    def fit(self, X, y, **fit_params):
        self.selected_features = cfs(X.toarray(), y)
        return self

    def transform(self, X, y=None, **fit_params):
        return X[:, self.selected_features]


class Terdchanakul2017(BaseEstimator, ClassifierMixin):
    """
    Reimplementation of approach by Terdchanakul et al. (2017): "Bug or Not? Bug Report Classification Using N-Gram IDF". In: Proceedings of the 33th IEEE International Conference on Software Maintenance and Evolution
    DOI: https://doi.org/10.1109/ICSME.2017.14
    """

    def __init__(self, clf, path_to_ngweight_exec, feature_selector=CorrelationBasedFeatureSelector()):
        """
        :param clf: Original paper used Logistic regression and random forest
        :param feature_selector: Either SelectKBest(chi2, k=10) (for training/testing setup) or
        CorrelationBasedFeatureSelector() (for cross validation)
        """

        self.clf = clf
        self.feature_selector = feature_selector
        self.path_to_ngweight_exec = path_to_ngweight_exec

    def fit(self, X, y):
        with tempfile.NamedTemporaryFile() as tmp_file:
            i = 0
            for row in X:
                tmp_file.write(b'\x02')
                tmp_file.write(str(i).encode())
                tmp_file.write(b'\x03')
                tmp_file.write(b'\n')
                tmp_file.write(b'\n')
                tmp_file.write(row.encode())
                tmp_file.write(b'\n')
                tmp_file.write(b'\n')
                i += 1

            process = subprocess.check_output(
                self.path_to_ngweight_exec + ' -w -s 0 < ' + tmp_file.name,
                shell=True
            )

        n_gram_dictionary = set([])
        max_n_gram = 1
        for line in process.splitlines():
            result = line.decode('utf-8').split('\t')

            # Filter out ngrams that only occur in one document
            if int(result[2]) != 1:
                n_gram_size = len(result[5].strip().split(" "))
                if max_n_gram < n_gram_size:
                    max_n_gram = n_gram_size

                if result[5].strip() != '':
                    n_gram_dictionary.add(result[5].strip())

        self.text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, max_n_gram), stop_words=None, token_pattern=r"(?u)\b\w+\b",
                                     vocabulary=n_gram_dictionary)),
            ('feature_selector', self.feature_selector),
            ('clf', self.clf)
        ])

        self.text_clf.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.text_clf.predict(X)

    def filter(self, df):
        df['combined'] = df['title'] + " " + df['description']
        df['combined'] = df['combined'].map(lambda x: remove_programming_characters(x))
        return df['combined']