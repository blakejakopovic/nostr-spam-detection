# Nostr Spam Detection

An experiment in building a machine learning model to label Nostr spam content for filtering and relay rejection.

## Dataset
`spam_202302101016_training.csv` (75,541) and `spam_202302101016_testing.csv` (10,000) contain a combined 85,541 labelled Nostr event contents values (72,522 ham and 13,019 spam). The data is biased by volume with asian language spam, and english language ham (non-spam) rows - however it has performed well in testing against recent Nostr events. The dataset was best-effort aggregated and largely labelled manually. It contains reviewed events flagged by Nostr event kind=1984, which indicate spam and other undesirable content. I've left the datasets raw to allow others to normalise themselves.

One final note. The dataset it likely pretty biased toward the past couple weeks of Nostr spam, and is less well rounded. You could train your models using another base spam detection set, or add more training as more Nostr spam labeled data is available.

## Results

Using a Naive Bayes classifier with the dataset, we were able to achieve 97%+ accuracy. Below is an example based on around 3,500 test records. The model size is around 25Mb. I've omitted a code example in this repo as I initially used Rust.

```
total: 3541 - tpos: 503 fpos: 67 tneg: 2946 fneg: 25
```

With further testing using more complex modelling, we were able to get upwards of 98% accuracy for spam detection using FastAI. The model is substantially larger however being around 160Mb.

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

# The notebook should run through cleanly, with each code segment being run once.

```

I've included a minimal Python Flask REST API endpoint that loads your model and can be called to get a spam score for an event (or just the event content).

```
# Start the Python Flask app
flask run

# To test the API
curl -X POST 'http://127.0.0.1:5000/test' --header 'Content-Type: application/json' --data-raw '{"content" : "Hello, is this spam or ham?"}'


# {"label": "ham", "score": "0.9913"}
```
