# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:21:50 2022

@author: Si Kemal
"""

import os
import pandas as pd
import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.layers import Bidirectional, SpatialDropout1D
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

CSV_URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"


#%% Step 1) Data Loading

df = pd.read_csv(CSV_URL)

# df_copy = df.copy() # backup
# df = df_copy.copy()

#%% Step 2) Data Inspection

df.head(10)
df.tail(10)
df.info()
df.describe()

df['sentiment'].unique() # to get the unique target values
df['review'][0]
df['sentiment'][0]

df.duplicated().sum() # 418 duplicated data

# <br /> HTML tags have to be 
# Numbers can be filtered
# Need to remove duplicated data


#%% Step 3) Data Cleaning

df = df.drop_duplicates()

# remove HTML tags

'<br />khgfvhkeafjfkh<br />'.replace('<br />', ' ')

review = df['review'].values
sentiment = df['sentiment'].values

for index,rev in enumerate(review):
    # remove html tags
    # ? dont be greedy
    # * zero or more occurences
    # Any character except new line (\n)
    review[index] = re.sub('<.*?>',' ',rev)
    
    # convert into lower case
    # remove numbers
    # ^ means NOT
    review[index] = re.sub('[^a-zA-Z]',' ',rev).lower().split()


#%% Step 4) Features Selection

# Nothing to select

#%% Step 5) Data Preprocessing
#           1) Convert into lower case

# Already done

#           2) Tokenization
vocab_size = 10000
oov_token = 'OOV'

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(review) # to learn all of the words
word_index = tokenizer.word_index
print(word_index)

train_sequences = tokenizer.texts_to_sequences(review) # to convert into numbers

#           3) Padding & truncating
length_of_review = [len(i) for i in train_sequences] # list comprehension
np.median(length_of_review) # to get the number of max length for padding

max_len = 180

padded_review = pad_sequences(train_sequences,maxlen=max_len,
                              padding='post',
                              truncating='post')

#           4) One Hot Encoding for the target

ohe = OneHotEncoder(sparse=False)
sentiment = ohe.fit_transform(np.expand_dims(sentiment,axis=-1))


#           5) Train test split

X_train,X_test,y_train,y_test = train_test_split(padded_review,sentiment,
                                                 test_size=0.3,
                                                 random_state=123)

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)


#%% Model Development

# Use LSTM layers, dropout, dense, input
# achieve >90% f1 score

embed_dims = 64
model=Sequential()
model.add(Input(shape=(180))) #input_length #features 
model.add(Embedding(vocab_size,output_dim=embed_dims))
model.add(Bidirectional(LSTM(128,return_sequences=True))) # only once return_se=true when LSTM meet LSTM after
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()

plot_model(model)
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['acc'])

hist = model.fit(X_train,y_train,
          epochs=3,
          batch_size=128,
          validation_data=(X_test,y_test))

#%%
import matplotlib.pyplot as plt

hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'],'r--',label='Training loss')
plt.plot(hist.history['val_loss'],label='Validation loss')
plt.legend()
plt.plot()

plt.figure()
plt.plot(hist.history['acc'],'r--',label='Training acc')
plt.plot(hist.history['val_acc'],label='Validation acc')
plt.legend()
plt.plot()

#%% Model Evaluation

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay


results = model.evaluate(X_test,y_test)
print(results)

pred_y = np.argmax(model.predict(X_test), axis=1)
true_y = np.argmax(y_test,axis=1)

cm = confusion_matrix(true_y,pred_y)
cr = classification_report(true_y,pred_y)
print(cm)
print(cr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%% Model Saving

MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
model.save(MODEL_SAVE_PATH)

import json

token_json = tokenizer.to_json()

TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)

OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)



