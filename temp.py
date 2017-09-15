import os
import numpy as np
import pandas as pd
import csv
from keras.models import Sequential, load_model, Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, Embedding, Input, Concatenate
from keras.layers.normalization import BatchNormalization
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import sys
sys.setrecursionlimit(2000)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import GRU

words = 256
train_file = 'train_split.csv'
test_file = 'test_split.csv'

def read_file(filename):
    data = pd.read_csv(filename)
    pos_comments = data["commentPositive"].fillna("")
    neg_comments = data["commentNegative"].fillna("")
    comments = data["comment"]
    labels = [(float(i) - 1) / 4 for i in data["reting"]]
    return comments, pos_comments, neg_comments, np.stack(labels)

X_train, pX_train, nX_train, y_train = read_file(train_file)
X_test, pX_test, nX_test, y_test = read_file(test_file)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train + pX_train + nX_train)
sequences = tokenizer.texts_to_sequences(X_train)
word_index = tokenizer.word_index
X_train = pad_sequences(sequences, maxlen = words)

pos_sequences = tokenizer.texts_to_sequences(pX_train)
pX_train = pad_sequences(pos_sequences, maxlen = words)

neg_sequences = tokenizer.texts_to_sequences(nX_train)
nX_train = pad_sequences(neg_sequences, maxlen = words)

s = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(s, maxlen=words)

s = tokenizer.texts_to_sequences(pX_test)
pX_test = pad_sequences(s, maxlen=words)

s = tokenizer.texts_to_sequences(nX_test)
nX_test = pad_sequences(s, maxlen=words)

embedding_layer = Embedding(len(word_index) + 1, 50, input_length=words)

sequence_input = Input(shape=(words, ), dtype='float32')
embedded_sequences = embedding_layer(sequence_input)

pos_sequence_input = Input(shape=(words, ), dtype='float32')
pos_embedded_sequences = embedding_layer(pos_sequence_input)

neg_sequence_input = Input(shape=(words, ), dtype='float32')
neg_embedded_sequences = embedding_layer(neg_sequence_input)

x = Concatenate()([embedded_sequences, pos_embedded_sequences, neg_embedded_sequences])
x = Dropout(0.9)(x)
x = Conv1D(256, 4, activation='relu', padding='same')(x)
x = MaxPooling1D(4)(x)
x = Conv1D(256, 4, activation='relu', padding='same')(x)
x = MaxPooling1D(4)(x)
x = Conv1D(256, 4, activation='relu', padding='same')(x)
x = MaxPooling1D(4)(x)
x = Conv1D(256, 4, activation='relu', padding='same')(x)
x = Flatten()(x)
x = Dropout(0.9)(x)
x = Dense(64, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[sequence_input, pos_sequence_input, neg_sequence_input], outputs=x)
print(model.summary())
callbacks = [ ModelCheckpoint('model.hdf5', monitor='val_loss', save_best_only=True) ]

model.compile(loss='mse', optimizer='adam')
model.fit([X_train, pX_train, nX_train], y_train, validation_data=([X_test, pX_test, nX_test],  y_test), epochs=500, batch_size=512, callbacks=callbacks)
