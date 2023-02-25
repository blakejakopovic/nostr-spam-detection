from flask import Flask
from flask import request
import json
import waitress
import os
from fastai.text.all import *

app = Flask(__name__)

learn = load_learner('models/awd_lstm_fully_trained_export')

@app.post("/test")
def test():
  if request.is_json:
    event = request.get_json()

    content = event["content"]

    prediction = learn.predict(content)

    label = prediction[0]
    score = "{:.4f}".format(prediction[2][prediction[1]].item())

    return json.dumps({'label': label, 'score': score}), 200

@app.post("/spam_score")
def spam_score():
  if request.is_json:
    event = request.get_json()

    content = event["content"]

    prediction = learn.predict(content)

    # The index for the spam (score) tensor is 1
    spam_score = "{:.4f}".format(prediction[2][1].item())

    return json.dumps({'spam_score': spam_score}), 200

if __name__ == "__main__":
     app.debug = False
     port = int(os.environ.get('PORT', 5000))
     waitress.serve(app, port=port)
