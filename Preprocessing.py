import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


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