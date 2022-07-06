import os
from bleach import clean
import streamlit as st
import numpy as np
import torch
from tweet_cleaner import *
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
category2score_map = {"Negative": -1, "Neutral": 0, "Positive": 1}


@st.cache
def load_model(path):
    model = AutoModelForSequenceClassification.from_pretrained(
        path)
    return model


tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest")

sentiment_model = load_model(model_path)

tweet_sentiment = pipeline('sentiment-analysis',
                           model=sentiment_model,
                           tokenizer=tokenizer,
                           framework="pt",
                           return_all_scores=True
                           )

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sentiment_standardizer(sentiment_labels, sentiment_scores):
    """
    Ensures sentiment score and label are consistent,
    i.e. coerces neutral tweet to have -0.5 < score < 0.2
    This is because sentiment score is a weighted average of probabilities,
    not something generted by the model itself
    sentiments: list
    scores: list
    Returns: standardized_score (list)
    """
    standardized_scores = [min(max(-0.5, score), 0.2) if sentiment ==
                           'Neutral' else score for sentiment, score in
                           zip(sentiment_labels, sentiment_scores)]
    return standardized_scores


def sentiment_calculator(sentiment_scores):
    sentiment_labels = []
    for score in sentiment_scores:
        if score < -0.5:
            sentiment_labels.append("Negative")
        elif score >= -0.5 and score < 0.2:
            sentiment_labels.append("Neutral")
        else:
            sentiment_labels.append("Positive")
    return sentiment_labels


def predict_sentiment(tweet):
    """
    tweet: list of strings or single string
    model: sentiment model object
    tokenizer: sentiment model tokenizer object
    """
    predicted_sentiments = tweet_sentiment(tweet)
    n_tweets = len(predicted_sentiments)

    results = [{predicted_sentiments[i][label]["label"]:
                predicted_sentiments[i][label]["score"]
                for label in np.arange(3)}
               for i in np.arange(n_tweets)]

    sentiment_scores = [sum([results[i][k] * v for
                             k, v in category2score_map.items()])
                        for i in np.arange(n_tweets)]
    sentiment_labels = sentiment_calculator(sentiment_scores)

    return sentiment_scores, sentiment_labels


def clean_tweet(tweet):
    tweet = cleaning(tweet)
    tweet = tweet_element_remover(tweet)
    return tweet


st.title('ADAPT - Social Media Analytics Demo')
st.markdown("""Twitter-RoBERTa Sentiment Analysis model""")
starting_text = st.text_area('Type in query...')

if starting_text:

    with st.spinner('Running model...'):
        cleaned_text = clean_tweet(starting_text)
        sentiments, scores = predict_sentiment(
            cleaned_text)

    st.markdown(f"""
    Cleaned text: {cleaned_text}

    Sentiment: {sentiments[0]}

    Score: {scores[0]}

    Negative: -1 to -0.5

    Neutral: -0.5 to 0.2

    Positive: 0.2 to 1
    """)
