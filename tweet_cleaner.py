import re
from airline_twitter import *

# Cleaning


def cleaning(tweet):
    """
    Operates one tweet at a time
    Cleans tweets from raw for pre-processing and display
    tweet: string
    """
    tweet = str(tweet)

    # "" handling
    tweet = tweet.replace("â€™", "'")
    tweet = tweet.replace("â€\x9d", "'")
    tweet = tweet.replace("â€œ", "'")
    tweet = tweet.replace("â€˜", "'")
    tweet = tweet.replace("Ã¼", "u")

    # Replace space characters with " "
    tweet = re.sub('_x000D_', ' ', tweet)
    tweet = re.sub('Â\xa0', ' ', tweet)
    tweet = re.sub('\\n', ' ', tweet)

    # Remove &lt; &gt; &le; &ge;
    tweet = re.sub('&lt;', '', tweet)
    tweet = re.sub('&gt;', '', tweet)
    tweet = re.sub('&le;', '', tweet)
    tweet = re.sub('&ge;', '', tweet)

    # Special character handling
    tweet = re.sub(r'andquot;', '', tweet)
    tweet = re.sub(r'[\S\w]*ð[\S\w]*', '', tweet)
    tweet = re.sub(r'[\S\w]*Ø[\S\w]*', '', tweet)
    tweet = re.sub(r'[\S\w]*(‡|±|˜|€|œ)[\S\w]*', '', tweet)
    tweet = re.sub("&amp;", "and", tweet)

    # Remove non-ASCII characters
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)

    # Long tweets exceeding max_seq_length, 1% of all tweets exceed 106 words
    if len(tweet.split(" ")) > 200:
        tweet = " ".join(tweet.split(" ")[:200])

    # Strip multiple whitespaces
    tweet = re.sub(' +', ' ', tweet)

    # Remove leading whitespace
    tweet = tweet.lstrip()

    return tweet


def tweet_element_remover(tweet):
    """
    Also operates one tweet at a time
    Removes tweet elements (hashtags, RTs, URLs) for better inference performance, not displayed
    tweet: String
    """

    tweet = tweet.lower()

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U0001F1F2-\U0001F1F4"  # Macau flag
                               u"\U0001F1E6-\U0001F1FF"  # flags
                               u"\U0001F600-\U0001F64F"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U0001F1F2"
                               u"\U0001F1F4"
                               u"\U0001F620"
                               u"\u200d"
                               u"\u2640-\u2642"
                               "]+", flags=re.UNICODE)

    # Remove emojis
    tweet = emoji_pattern.sub(r'', tweet)

    tweet = re.sub(r'-', " ", tweet)

    # Remove URL links
    tweet = re.sub('http\S+', '', tweet)

    # Remove users
    tweet = re.sub("(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)", "", tweet)

    # Remove # characters, keep words
    tweet = re.sub('#', ' ', tweet)

    # Strip multiple whitespaces
    tweet = re.sub(' +', ' ', tweet)

    # Remove leading whitespace
    tweet = tweet.lstrip()

    return tweet
