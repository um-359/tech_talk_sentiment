from sentiment import *
from tweet_cleaner import *
import streamlit as st
# import st_state_patch
# import SessionState
import numpy as np
import os

sentiment_model = Sentiment(
    f"cardiffnlp/twitter-roberta-base-sentiment-latest")

if __name__ == "__main__":

    print(os.listdir())

    st.title('ADAPT - Social Media Analytics Demo')
    st.markdown("""Tweet Classification model""")
    starting_text = st.text_area('Type in query...')

    if starting_text:

        with st.spinner('Running model...'):
            sentiments, scores = sentiment_model.predict_sentiment(
                starting_text)

        st.markdown(f"""
        Sentiment: {sentiments[0]}

        Score: {scores[0]}
        """)
