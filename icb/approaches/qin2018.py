import re
import nltk

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras_preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, ClassifierMixin


class Qin2018(BaseEstimator, ClassifierMixin):
    """
    Reimplementation of approach by Qin et al. (2018): "Classifying Bug Reports into Bugs and Non-bugs Using LSTM". In: Proceedings of the 10th Asia-Pacific Symposium on Internetware
    DOI: https://doi.org/10.1145/3275219.3275239
    """

    # Max number of words in each issue.
    MAX_SEQUENCE_LENGTH = 100

    # This is fixed.
    EMBEDDING_DIM = 100

    def __init__(self, epochs=5, batch_size=64):
        self.word_to_int = {}
        self.input_length = 100
        self.embedding_length = 826900
        self.epochs = epochs
        self.batch_size = batch_size

    def create_model(self):
        model = Sequential()
        model.add(Embedding(self.embedding_length+1, self.EMBEDDING_DIM, input_length=self.input_length))
        model.add(LSTM(100, dropout=0.5, recurrent_dropout=0.5))
        model.add(Dense(1, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        # First, we need to get our dictionary that maps all words that occur to numbers
        for issue in X.values:
            raw_text = self.preprocess(issue)
            i = 1
            for word in raw_text.split(" "):
                if word not in self.word_to_int.keys():
                    self.word_to_int[word] = i
                    i += 1

        # Second, we need to go through all issue reports and map their words to the numbers according to the dictionary
        data = []
        for issue in X.values:
            data.append([self.word_to_int[word] for word in self.preprocess(issue).split(" ")])

        # Third, we pad sequences to a length of 100 according to the paper
        X = pad_sequences(data, maxlen=self.MAX_SEQUENCE_LENGTH)

        # Create classifier
        self.input_length = X.shape[1]
        self.embedding_length = len(self.word_to_int.values())
        self.clf = KerasClassifier(build_fn=self.create_model, epochs=self.epochs, batch_size=self.batch_size)
        self.clf.fit(X, y)

        return self

    def predict(self, X, y=None):

        # We need to map the issues with their words to the corresponding integer numbers from our trained mapping
        data = []
        for issue in X.values:
            mapping = []
            for word in self.preprocess(issue).split(" "):
                # Append 0 if word is not found in our embedding vector
                if word not in self.word_to_int:
                    mapping.append(0)
                else:
                    mapping.append(self.word_to_int[word])
            data.append(mapping)

        # We pad sequences to a length of 100 according to the paper
        X = pad_sequences(data, maxlen=self.MAX_SEQUENCE_LENGTH)
        return self.clf.predict(X)

    def filter(self, df):
        return df['title'] + " " + df['description']

    def preprocess(self, all_text):
        # Lowercasing not mentioned in paper
        all_text = all_text.lower()
        tokens = nltk.word_tokenize(all_text)
        joined_tokens = " ".join(tokens)
        return re.sub(r"[0-9]+", "<NUM>", joined_tokens)