# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:52:09 2022

@author: Si Kemal
"""

#%% Deployment
import os
import numpy as np
import re
import json
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# 1) Trained model --> loading from .h5
# 2) tokenizer ---> loading from json
# 3) MMS/OHE ---> loading from pickle

TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')

# to load trained model
loaded_model = load_model(os.path.join(os.getcwd(),'model.h5'))

loaded_model.summary()

# to load tokenizer
with open(TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)
    
with open(OHE_PATH,'rb') as file:
    loaded_ohe = pickle.load(file)

#%%
input_review = 'This movie so good, the trailer intrigues me to watch.\
                    The movie is funny. I love it so much'

# preprocessing

input_review = re.sub('<.*?',' ',input_review)
input_review = re.sub('[^a-zA-Z]',' ',input_review).lower().split()

tokenizer = tokenizer_from_json(loaded_tokenizer)
input_review_encoded = tokenizer.texts_to_sequences(input_review)

input_review_encoded = pad_sequences(np.array(input_review_encoded).T,
                                     maxlen=180,
                                     padding='post',
                                     truncating='post')


outcome = loaded_model.predict(np.expand_dims(input_review_encoded,axis=-1))

print(loaded_ohe.inverse_transform(outcome))