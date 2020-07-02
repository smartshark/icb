import re
import nltk

from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
import pandas as pd
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
from tempfile import NamedTemporaryFile
import os.path as path

# Code partially re-used from replication kit:
# https://github.com/danaderp/SecureReqNet/

class Embeddings:

    def __init__(self):
        self.__wpt = nltk.WordPunctTokenizer()
        self.__stop_words = nltk.corpus.stopwords.words('english')
        self.__remove_terms = punctuation + '0123456789'

    def __split_camel_case_token(self, token):
        return re.sub('([a-z])([A-Z])', r'\1 \2', token).split()

    def __clean_punctuation(self, token):
        remove_terms = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
        cleaned = token
        for p in remove_terms:
            cleaned = cleaned.replace(p, ' ')
        return cleaned.split()

    def __clean(self, token):
        to_return = self.__clean_punctuation(token)
        new_tokens = []
        for t in to_return:
            new_tokens += self.__split_camel_case_token(t)
        to_return = new_tokens
        return to_return

    def __normalize_document(self, doc):
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
        doc = doc.lower()
        doc = doc.strip()
        # tokenize document
        tokens = self.__wpt.tokenize(doc)
        # Filter stopwords out of document
        filtered_tokens = [token for token in tokens if token not in self.__stop_words]
        # Filtering Stemmings
        filtered_tokens = [SnowballStemmer("english").stem(token) for token in filtered_tokens]
        # Filtering remove-terms
        filtered_tokens = [token for token in filtered_tokens if token not in self.__remove_terms and len(token) > 2]
        # re-create document from filtered tokens
        return filtered_tokens

    def preprocess(self, sentence, vocab_set=None):
        tokens = sentence.split()
        new_tokens = []
        for token in tokens:
            new_tokens += self.__clean(token)
        tokens = new_tokens

        tokens = self.__normalize_document(' '.join(tokens))

        return tokens

    def get_embeddings_dict(self, embeddings_filename):
        embeddings_df = pd.read_csv(embeddings_filename)
        embeddings_dict = dict()
        for col in list(embeddings_df)[1:]:
            embeddings_dict[col] = list(embeddings_df[col])
        return embeddings_dict

    def vectorize(self, sentence, embeddings_dict):
        processed_sentence = self.preprocess(sentence)

        matrix = []
        for token in processed_sentence:
            if token in embeddings_dict:
                matrix.insert(0, embeddings_dict[token])
        return np.matrix(matrix)


class Palacio2019(BaseEstimator, ClassifierMixin):
    """
    Reimplementation of approach by Palacio et al. (2018): "Learning to Identify Security-Related Issues Using
    Convolutional Neural Networks". In: Proceedings of the 2019 IEEE International Conference on Software Maintenance
    and Evolution (ICSME)
    DOI: 10.1109/ICSME.2019.00024
    """

    EMBEDDING_PRETRAIN_MAX_WORDS = 5000
    N_FILTERS = 128
    N_CLASSES = 2

    def __init__(self, epochs=2000):
        self.epochs = epochs
        # We use the maximum lengths of docs from the embedding here. this is highest possible value in the original
        # paper. The drawback is that we require more time for training, the advantage is that we do not need to know
        # about the test data.
        self.max_len_sentences = self.EMBEDDING_PRETRAIN_MAX_WORDS

    def create_model(self):
        # configure neural network
        input_sh = (self.max_len_sentences, self.embed_size, 1)
        gram_input = Input(shape=input_sh)

        # First Layer: Three parallel n-grams convolutions
        conv_filter_1_gram = Conv2D(filters=self.N_FILTERS, input_shape=input_sh, activation='relu',
                                    kernel_size=(1, self.embed_size), padding='valid', data_format="channels_last")(
            gram_input)
        conv_filter_3_gram = Conv2D(filters=self.N_FILTERS, input_shape=input_sh, activation='relu',
                                    kernel_size=(3, self.embed_size), padding='valid')(gram_input)
        conv_filter_5_gram = Conv2D(filters=self.N_FILTERS, input_shape=input_sh, activation='relu',
                                    kernel_size=(5, self.embed_size), padding='valid')(gram_input)

        # Second Layer: Three parallel maxpoolings
        max_pool_1_gram = MaxPooling2D(pool_size=((self.max_len_sentences - 1 + 1), 1), strides=None, padding='valid')(
            conv_filter_1_gram)
        max_pool_3_gram = MaxPooling2D(pool_size=((self.max_len_sentences - 3 + 1), 1), strides=None, padding='valid')(
            conv_filter_3_gram)
        max_pool_5_gram = MaxPooling2D(pool_size=((self.max_len_sentences - 5 + 1), 1), strides=None, padding='valid')(
            conv_filter_5_gram)

        # Third Layer: Three parallel fully connected layer
        fully_connected_1_gram = Flatten()(max_pool_1_gram)
        fully_connected_3_gram = Flatten()(max_pool_3_gram)
        fully_connected_5_gram = Flatten()(max_pool_5_gram)

        # Fourth Layer: Merge parallel layers with a dropout layer
        merged_vector = concatenate([fully_connected_1_gram, fully_connected_3_gram,
                                     fully_connected_5_gram], axis=-1)
        integration_layer = Dropout(0.2)(merged_vector)

        # Output layer: softmax to determine class
        predictions = Dense(self.N_CLASSES, activation='softmax')(integration_layer)
        keras_model = Model(inputs=[gram_input], outputs=[predictions])
        keras_model.compile(optimizer='adam', loss='binary_crossentropy',
                            metrics=['accuracy'])
        return keras_model

    def fit(self, X, y):
        # compute embeddings
        embeddings = Embeddings()
        pre_corpora = [doc for doc in X if len(doc) < self.EMBEDDING_PRETRAIN_MAX_WORDS]
        embed_path = 'data/word_embeddings-embed_size_100-epochs_100.csv'
        embeddings_dict = embeddings.get_embeddings_dict(embed_path)
        corpora = [embeddings.vectorize(doc, embeddings_dict) for doc in pre_corpora]
        self.embed_size = np.size(corpora[0][0])

        # reshape data for training
        shape_x = (len(corpora), self.max_len_sentences, self.embed_size, 1)
        with NamedTemporaryFile() as ntf:
            corpora_x = np.memmap(ntf, dtype='float32', mode='w+', shape=shape_x)
            for doc in range(len(corpora)):
                # print(corpora_train[doc].shape[1])
                for words_rows in range(corpora[doc].shape[0]):
                    embed_flatten = np.array(corpora[doc][words_rows]).flatten()
                    for embedding_cols in range(embed_flatten.shape[0]):
                        corpora_x[doc, words_rows, embedding_cols, 0] = embed_flatten[embedding_cols]

            target_y = np.array([[val, abs(val - 1)] for doc, val in zip(X,y)
                                 if len(doc)<self.EMBEDDING_PRETRAIN_MAX_WORDS])

            # we use patience=3 instead of 100 because we never observed that there was an increase after convergence, even
            # with patience=100
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
            callbacks_list = [es]
            self.clf = KerasClassifier(build_fn=self.create_model, epochs=self.epochs,
                                       validation_split=0.2, callbacks=callbacks_list)
            self.clf.fit(corpora_x, target_y)

        return self

    def predict(self, X, y=None):
        # compute embeddings
        embeddings = Embeddings()
        pre_corpora = [doc for doc in X]
        embed_path = 'data/word_embeddings-embed_size_100-epochs_100.csv'
        embeddings_dict = embeddings.get_embeddings_dict(embed_path)
        corpora = [embeddings.vectorize(doc, embeddings_dict) for doc in pre_corpora]

        # reshape data for prediction
        shape_x = (len(corpora), self.max_len_sentences, self.embed_size, 1)
        with NamedTemporaryFile() as ntf:
            corpora_x = np.memmap(ntf, dtype='float32', mode='w+', shape=shape_x)
            for doc in range(len(corpora)):
                # print(corpora_train[doc].shape[1])
                for words_rows in range(corpora[doc].shape[0]):
                    if words_rows < self.max_len_sentences:  # cutoff for very long documents
                        embed_flatten = np.array(corpora[doc][words_rows]).flatten()  # <--- Capture doc and word
                        for embedding_cols in range(embed_flatten.shape[0]):
                            corpora_x[doc, words_rows, embedding_cols, 0] = embed_flatten[embedding_cols]
            return self.clf.predict(corpora_x)

    def filter(self, df):
        return df['description']

