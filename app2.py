from flask import Flask
from flask import request
import json
import waitress
import pickle
import os

app = Flask(__name__)

filename = 'models/MultinomialNB.sav'
(model, vectorizer) = pickle.load(open(filename, 'rb'))

@app.post("/test")
def test():
  if request.is_json:
    event = request.get_json()

    content = event["content"]

    prediction = model.predict_proba(vectorizer.transform([content]))[0]

    # Note: Prediction is [ham_score, spam_score]
    if prediction[0] > prediction[1]:
      label = "ham"
      score = "{:.4f}".format(prediction[0])
    else:
      label = "spam"
      score = "{:.4f}".format(prediction[1])

    return json.dumps({'label': label, 'score': score}), 200

@app.post("/spam_score")
def spam_score():
  if request.is_json:
    event = request.get_json()

    content = event["content"]

    prediction = model.predict_proba(vectorizer.transform([content]))[0]

    # The index for the spam (score) tensor is 1
    spam_score = "{:.4f}".format(prediction[1])

    return json.dumps({'spam_score': spam_score}), 200

if __name__ == "__main__":
     app.debug = False
     port = int(os.environ.get('PORT', 50011))
     waitress.serve(app, port=port)
