import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.feature_extraction.text import CountVectorizer


class Chawla2015(BaseEstimator, ClassifierMixin):
    """
    Reimplementation of approach by Chawla et al. (2015): "An automated approach for bug categorization using fuzzy logic." In: Proceedings of the 8th India Software Engineering Conference.
    DOI: https://doi.org/10.1145/2723742.2723751
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        vect = CountVectorizer(stop_words='english')
        X = vect.fit_transform(X)
        df2 = pd.DataFrame(X.toarray())
        df2['classification'] = y
        true_mask = df2['classification'] == 1
        colsums = df2.sum()
        colsums_true = df2.loc[true_mask].sum()
        colsums_false = df2.loc[~true_mask].sum()
        self.term_memberships = {}
        for column, value in df2.iteritems():
            if column != 'classification':
                #assert sum(df2.loc[df2['classification'] == 0].iloc[:, column]) == colsums_false.iat[column]
                #assert sum(df2.loc[df2['classification'] == 1].iloc[:, column]) == colsums_true.iat[column]
                #assert sum(df2.iloc[:, column]) == colsums.iat[column]
                #print("foo7")
                #self.term_memberships[vect.get_feature_names()[column]] = (
                #    sum(df2.loc[df2['classification'] == 0].iloc[:, column]) / sum(df2.iloc[:, column]),
                #    sum(df2.loc[df2['classification'] == 1].iloc[:, column]) / sum(df2.iloc[:, column])
                #)
                self.term_memberships[vect.get_feature_names()[column]] = (
                    colsums_false.iat[column] / colsums.iat[column],
                    colsums_true.iat[column] / colsums.iat[column]
                )

        return self

    def predict(self, X, y=None):
        results = []
        for row in X:
            # Make use of the countvectorizer for all preprocessing steps
            vect = CountVectorizer(stop_words='english')
            try:
                vect.fit_transform([row])
            except ValueError:
                results.append(0)
                continue

            # Get the issue reports membership score
            sum_non_bug = 1.0
            sum_bug = 1.0
            for feature in vect.get_feature_names():
                try:
                    sum_non_bug *= (1 - self.term_memberships[feature][0])
                    sum_bug *= (1 - self.term_memberships[feature][1])
                except KeyError:
                    # If the term is not previously seen we do not have data about it
                    pass

            # If their score is bigger for the bug terms, assign 1, otherwise 0
            if (1-sum_bug) > (1-sum_non_bug):
                results.append(1)
            else:
                results.append(0)

        return results

    def score(self, X, y=None, **kwargs):
        results = self.predict(X)

        # Count number of differences between results and real value and divide by number of observations
        differences = (sum(i != j for i, j in zip(results, y))) / len(results)

        # The bigger the better, thats why we need to calculate 1-differences
        return 1-differences

    def filter(self, df):
        return df.title
