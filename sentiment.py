import os
import numpy as np
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Sentiment:
    def __init__(self, model_path):
        self.sentiment_mapping = {"LABEL_0": "Negative",
                                  "LABEL_1": "Neutral",
                                  "LABEL_2": "Positive"}
        self.category2score_map = {"Negative": -1, "Neutral": 0, "Positive": 1}
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tweet_sentiment = pipeline('sentiment-analysis',
                                        model=self.model,
                                        tokenizer=self.tokenizer,
                                        framework="pt",
                                        return_all_scores=True
                                        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def sentiment_standardizer(self, sentiments, scores):
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
                               'Neutral' else score for
                               sentiment, score in zip(sentiments, scores)]
        return standardized_scores

    def predict_sentiment(self, tweet):
        """
        tweet: list of strings or single string
        model: sentiment model object
        tokenizer: sentiment model tokenizer object
        """
        predicted_sentiments = self.tweet_sentiment(tweet)
        n_tweets = len(predicted_sentiments)

        results = [{predicted_sentiments[i][label]["label"]:
                    predicted_sentiments[i][label]["score"]
                    for label in np.arange(3)} for i in np.arange(n_tweets)]

        scores = [sum([results[i][k] * v for
                       k, v in self.category2score_map.items()])
                  for i in np.arange(n_tweets)]
        sentiments = [max(results[i], key=results[i].get)
                      for i in np.arange(n_tweets)]
        scores = self.sentiment_standardizer(sentiments, scores)

        return sentiments, scores
