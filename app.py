from flask import Flask
from flask import request
import json
import numpy as np
from fastai.text.all import *

app = Flask(__name__)
learn = load_learner('models/awd_lstm_fully_trained_export')

@app.post("/test")
def test():
  if request.is_json:
    event = request.get_json()

    content = event["content"]

    prediction = learn.predict(content)

    response = {}
    response['label'] = prediction[0]
    response['score'] = "{:.4f}".format(prediction[2][prediction[1]].item())

    return json.dumps(response), 200
