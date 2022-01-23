import os
import sys
import string
import random
import json
import requests
import numpy as np
import tensorflow as tf
import pickle

from flask import Flask, request, redirect, url_for, render_template

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wtforms.validators import DataRequired

from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

"""
Constants
"""
MODEL_URI = 'http://localhost:8502/v1/models/emotion_model:predict'
OUTPUT_DIR = 'static'
SIZE = 128


"""
Utility functions
"""
#class TextForm(FlaskForm):
#    text = StringField('Text',
#                        validators=[DataRequired()])
#    submit = SubmitField('Submit text')



def tokenize(text):
  #  a tokenier to tokenize the words and create sequences of tokenized words
  with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequences, padding='post', maxlen=30)

  return padded

def get_prediction(uploaded_text):
    text = tokenize(uploaded_text)
    data = json.dumps({'instances': text.tolist()}) 
    response = requests.post(MODEL_URI, data=data.encode())
    result = json.loads(response.text)
    prediction = result['predictions'][0]
    predicted_class = np.argmax(prediction)#.astype('uint8')
    return prediction


"""
Routes
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        uploaded_text = request.form['text']
        result = get_prediction(uploaded_text)
        return render_template('show.html', result=result)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)