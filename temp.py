import os
import numpy as np
import csv
from keras.models import Sequential, load_model, Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, Embedding, Input, Concatenate, GRU
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

words = 375

pos_comments = []
neg_comments = []
comments = []
labels = []
neg_labels = []
pos_labels = []
brands = []

with open('X_train.csv') as X_train:
    next(X_train)
    reader = csv.reader(X_train)
    for row in reader:
        pos_comments.append(row[len(row) - 1])
        pos_labels.append(row[len(row) - 5])
        neg_comments.append(row[len(row) - 2])
        neg_labels.append(row[len(row) - 5])
        comments.append(row[len(row) - 3])
        labels.append((float(row[len(row) - 5]) - 1) / 4)
#        brands.append(row[len(row) - 8])

labels = np.stack(labels)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen = words)

pos_labels = np.stack(pos_labels)
pos_tokenizer = Tokenizer()
pos_tokenizer.fit_on_texts(pos_comments)
pos_sequences = tokenizer.texts_to_sequences(pos_comments)
pos_word_index = tokenizer.word_index
pos_data = pad_sequences(pos_sequences, maxlen = words)

neg_labels = np.stack(neg_labels)
neg_tokenizer = Tokenizer()
neg_tokenizer.fit_on_texts(neg_comments)
neg_sequences = tokenizer.texts_to_sequences(neg_comments)
neg_word_index = tokenizer.word_index
neg_data = pad_sequences(neg_sequences, maxlen = words)

nums = [i for i in range(len(data))]

nums_train, nums_test, y_train, y_test = train_test_split(nums, labels, test_size=0.1)

X_train = np.stack([data[i] for i in nums_train])
pX_train = np.stack([pos_data[i] for i in nums_train])
nX_train = np.stack([neg_data[i] for i in nums_train])
#bX_train = np.stack([brands[i] for i in nums_train])

X_test = np.stack([data[i] for i in nums_test])
pX_test = np.stack([pos_data[i] for i in nums_test])
nX_test = np.stack([neg_data[i] for i in nums_test])
#bX_test = np.stack([brands[i] for i in nums_test])

embedding_layer = Embedding(len(word_index) + 1, 50, input_length=words)
pos_embedding_layer = Embedding(len(pos_word_index) + 1, 50, input_length=words)
neg_embedding_layer = Embedding(len(neg_word_index) + 1, 50, input_length=words)

sequence_input = Input(shape=(words, ), dtype='float32')
embedded_sequences = embedding_layer(sequence_input)

pos_sequence_input = Input(shape=(words, ), dtype='float32')
pos_embedded_sequences = embedding_layer(pos_sequence_input)

neg_sequence_input = Input(shape=(words, ), dtype='float32')
neg_embedded_sequences = embedding_layer(neg_sequence_input)

#brand_input = Input(shape=(1, ), dtype='float32')
#brand_dense = Dense(1, activation='relu')(brand_input)

x = Concatenate()([embedded_sequences, pos_embedded_sequences, neg_embedded_sequences])
'''x = Conv1D(256, 5, activation='relu', padding='same')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(256, 5, activation='relu', padding='same')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(256, 5, activation='relu', padding='same')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(256, 5, activation='relu', padding='same')(x)'''
x = Conv1D(256, 5, activation='relu', padding='same')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(256, 5, activation='relu', padding='same')(x)
x = MaxPooling1D(5)(x)
x = Dropout(0.2)(x)
x = Conv1D(256, 5, activation='relu', padding='same')(x)
x = Dropout(0.2)(x)
x = Conv1D(256, 5, activation='relu', padding='same')(x)
x = Conv1D(256, 5, activation='relu', padding='same')(x)
x = MaxPooling1D(5)(x)
x = Flatten()(x)
#x = Concatenate()([x, brand_input])
x = Dense(16, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[sequence_input, pos_sequence_input, neg_sequence_input], outputs=x)
print(model.summary())
callbacks = [ ModelCheckpoint('model.hdf5', monitor='val_loss', save_best_only=True) ]

model.compile(loss='mse', optimizer='adam')
model.fit([X_train, pX_train, nX_train], y_train, validation_data=([X_test, pX_test, nX_test],  y_test), epochs=15, batch_size=512, callbacks=callbacks)

model = load_model('model.hdf5')
model.compile(loss='mse', optimizer=Adam(lr=0.0001))
model.fit([X_train, pX_train, nX_train], y_train, validation_data=([X_test, pX_test, nX_test],  y_test), epochs=100, batch_size=512, callbacks=callbacks)

#model = load_model('model.hdf5')
#model.compile(loss='mae', optimizer='rmsprop')
#model.fit([X_train, pX_train, nX_train], y_train, validation_data=([X_test, pX_test, nX_test],  y_test), epochs=10, batch_size=512, callbacks=callbacks)

'''a = model.predict([X_test, pX_test, nX_test], batch_size=128)
err = 0
j = 0
for i in range(len(a)):
    if abs(float(a[i]) - y_test[i]) > err:
        err = abs(a[i] - y_test[i])
        j = i'''
