# Nostr Spam Detection

An experiment in building a machine learning model to label Nostr spam content for filtering and relay rejection.

## Dataset
The latest dataset is `labelled_nostr_events_20230225000.csv` with 92,000 rows. It contains labelled nostr event content with either `spam` (bad - 13k) or `ham` (good - 79k). The data is biased by volume with asian language spam, and english language ham (non-spam) rows - however it has performed well in testing against recent Nostr events. The dataset was best-effort aggregated and largely labelled manually. It contains reviewed events flagged by Nostr event kind=1984, which indicate spam and other undesirable content. I've left the datasets raw to allow others to normalise themselves.

One final note. The dataset it likely pretty biased toward the past month of Nostr spam, and is less well rounded. You could train your models using another base spam detection set, or add more training as more Nostr spam labeled data is available.

## Results

Using a Naive Bayes classifier with the dataset, we are able to achieve 98.26%+ accuracy. The model size is around 22Mb, and takes less than a minute to train. To train this model, take a look at `Nostr MultinominalNB Spam Detection.ipynb`.

With more complex language modelling and deep learning, we were able to get upwards of 98% accuracy for spam detection using FastAI. The model is substantially larger however being around 160Mb, and takes significantly longer to train (can be hours on a laptop CPU).

## Dependencies
* Python
* Juypter
* FastAI

## Usage

First you will need to build your model. It can take over an hour to train using a CPU instead of GPU. Once you have the dependencies installed, you can run the following to start training your model.

```
git clone https://github.com/blakejakopovic/nostr-spam-detection
cd nostr-spam-detection

jupyter notebook

# The notebooks should run through cleanly, with each code segment being run once.

```

I've included a minimal Python Flask REST API endpoint that loads your model and can be called to get a spam score for an event (or just the event content).

```
# Start the prediction example apps
gunicorn --workers 5 --bind localhost:5000 app:app (app:app for FastAI or app2:app for the bayes model)

# To get a label and score for content
curl -X POST 'http://127.0.0.1:5000/test' --header 'Content-Type: application/json' --data-raw '{"content" : "Hello, is this spam or ham?"}'
# {"label": "ham", "score": "0.9913"}

# To get a spam_score directly for content
curl -X POST 'http://127.0.0.1:5000/spam_score' --header 'Content-Type: application/json' --data-raw '{"content" : "Hello, how spammy is this?"}'
# {"spam_score": "0.9913"}


```
