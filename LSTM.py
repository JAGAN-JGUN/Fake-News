import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

print(device_lib.list_local_devices())

MAX_SEQ_LEN = 1000
MAX_NB_WORD = 200000

#1
training = pd.read_csv('Data/train.csv')
print(training.columns)
print(training[0:5])
texts = []
labels = []

for i in range(training.text.shape[0]):
    text1 = training.title[i]
    text2 = training.text[i]
    text = str(text1) +""+ str(text2)
    texts.append(text)
    labels.append(training.label[i])
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

words_index = tokenizer.word_index
print('Found %s unique tokens.' % len(words_index))

data = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)
label = to_categorical(np.asarray(labels),num_classes = 2)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', label.shape)

#2
from sklearn.model_selection import train_test_split

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
label = label[indices]
x_train, x_test, y_train, y_test = train_test_split( data, label, test_size=0.20, random_state=42)
x_test, x_val, y_test, y_val = train_test_split( data, label, test_size=0.20, random_state=42)
print('Size of train, validation, test:', len(y_train), len(y_val), len(y_test))

print('real & fake news in train,valt,test:')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))
print(y_test.sum(axis=0))

#3
import os
from keras.layers import Embedding

EMBED_DIM = 100
GLOVE_DIR = "data" 
embeddings_index = {}
fping = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8")
for line in fping:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
fping.close()

print('Total %s word vectors in Glove.' % len(embeddings_index))

embed_matrix = np.random.random((len(words_index) + 1, EMBED_DIM))
for word, i in words_index.items():
    embed_vector = embeddings_index.get(word)
    if embed_vector is not None:
        embed_matrix[i] = embed_vector
        
embed_layer = Embedding(len(words_index) + 1,
                            EMBED_DIM,
                            weights=[embed_matrix],
                            input_length=MAX_SEQ_LEN)

from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import BatchNormalization
from keras.layers import Dense,Dropout, MaxPooling1D, Embedding, LSTM, GRU

embed_vecor_length = 32
model = Sequential()
model.add(embed_layer)
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs=5, batch_size=64)

model.save('lstm.h5')

embed_vecor_length = 32
modelg = Sequential()
modelg.add(embed_layer)
modelg.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
modelg.add(BatchNormalization())
modelg.add(Dense(2, activation='softmax'))
modelg.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
print(modelg.summary())
modelg.fit(x_train, y_train, epochs=2, batch_size=64)
modelg.save('gru.h5')

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
test_preds = model.predict(x_test)
test_preds = np.round(test_preds)
correct_prediction = float(sum(test_preds == y_test)[0])
print("Correct predictions:", correct_prediction)
print("Total number of test examples:", len(y_test))
print("Accuracy of model1: ", correct_prediction/float(len(y_test)))

x_pred = model.predict(x_test)
x_pred = np.round(x_pred)
x_pred = x_pred.argmax(1)
y_test_s = y_test.argmax(1)
cm = confusion_matrix(y_test_s, x_pred)
plt.matshow(cm, cmap=plt.cm.binary, interpolation='nearest',)
plt.title('Confusion matrix - LSTM')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.show()

x_pred = modelg.predict(x_test)
x_pred = np.round(x_pred)
x_pred = x_pred.argmax(1)
y_test_s = y_test.argmax(1)
cm = confusion_matrix(y_test_s, x_pred)
plt.matshow(cm, cmap=plt.cm.binary, interpolation='nearest',)
plt.title('Confusion matrix - GRU')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.show()