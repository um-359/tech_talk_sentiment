import streamlit as st
from tweet_cleaner import *
import os
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
category2score_map = {"Negative": -1, "Neutral": 0, "Positive": 1}


@st.cache
def load_model(path):
    model = AutoModelForSequenceClassification.from_pretrained(
        path)
    return model


tokenizer = AutoTokenizer.from_pretrained(
    f"cardiffnlp/twitter-roberta-base-sentiment-latest")

sentiment_model = load_model(model_path)

tweet_sentiment = pipeline('sentiment-analysis',
                           model=sentiment_model,
                           tokenizer=tokenizer,
                           framework="pt",
                           return_all_scores=True
                           )

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sentiment_standardizer(sentiments, scores):
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
                           'Neutral' else score for sentiment, score in zip(sentiments, scores)]
    return standardized_scores


def sentiment_calculator(scores):
    sentiments = []
    for score in scores:
        if score < -0.5:
            sentiments.append("Negative")
        elif score >= -0.5 and score < 0.2:
            sentiments.append("Neutral")
        else:
            sentiments.append("Positive")
    return sentiments


def predict_sentiment(tweet):
    """
    tweet: list of strings or single string
    model: sentiment model object
    tokenizer: sentiment model tokenizer object
    """
    predicted_sentiments = tweet_sentiment(tweet)
    n_tweets = len(predicted_sentiments)

    # results = [{self.sentiment_mapping[predicted_sentiments[i][label]["label"]]:
    #             predicted_sentiments[i][label]["score"] for label in np.arange(3)} for i in np.arange(n_tweets)]
    results = [{predicted_sentiments[i][label]["label"]:
                predicted_sentiments[i][label]["score"] for label in np.arange(3)}
               for i in np.arange(n_tweets)]

    scores = [sum([results[i][k] * v for k, v in category2score_map.items()])
              for i in np.arange(n_tweets)]
    sentiments = sentiment_calculator(scores)

    return sentiments, scores


st.title('ADAPT - Social Media Analytics Demo')
st.markdown("""Twitter-RoBERTa Sentiment Analysis model""")
starting_text = st.text_area('Type in query...')

if starting_text:

    with st.spinner('Running model...'):
        sentiments, scores = predict_sentiment(
            starting_text)

    st.markdown(f"""
    Sentiment: {sentiments[0]}

    Score: {scores[0]}

    Negative: -1 to -0.5

    Neutral: -0.5 to 0.2

    Positive: 0.2 to 1
    """)
