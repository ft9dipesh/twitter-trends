import nltk
import matplotlib.pyplot as plt
import pandas as pd

from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import utils


# Setup nltk libs
nltk.download("averaged_perceptron_tagger")
nltk.download("vader_lexicon")


def get_tags(tokens):
    return nltk.tag.pos_tag(tokens)


def plot_tags_freq(tags):
    tags_dict = {}
    for (k,v) in tags:
        if v in tags_dict:
            tags_dict[v] = tags_dict[v] + 1
        else:
            tags_dict[v] = 1

    sorted_dict = dict(sorted(
        tags_dict.items(), 
        key = lambda item: item[1], 
        reverse=True
    ))
    plt.figure(figsize=(15,5))
    fig, ax = plt.subplots()
    
    ax.bar(list(sorted_dict.keys()), sorted_dict.values())
    ax.set_yscale("log")
    
    ax.set_xlabel("POS Tags")
    ax.set_ylabel("log of counts")
    
    plt.tick_params(axis="x", which="major", labelsize=8)
    plt.savefig("results/pos-tag-frequency.png")


def sentiment_analyzer(tweets, num_tweets):
    positive = 0
    negative = 0
    neutral = 0
    polarity = 0
    tweet_list = []
    neutral_list = []
    negative_list = []
    positive_list = []

    for tweet in tweets:
        tweet_list.append(tweet.text)
        analysis = TextBlob(tweet.text)
        score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)

        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']

        polarity += analysis.sentiment.polarity

        if neg > pos:
            negative_list.append(tweet.text)
            negative += 1
        elif pos > neg:
            positive_list.append(tweet.text)
            positive += 1
        elif pos == neg:
            neutral_list.append(tweet.text)
            neutral += 1

    positive = utils.percentage(positive, num_tweets)
    negative = utils.percentage(negative, num_tweets)
    neutral = utils.percentage(neutral, num_tweets)
    polarity = utils.percentage(polarity, num_tweets)

    positive = format(positive, '.2f')
    negative = format(negative, '.2f')
    neutral = format(neutral, '.2f')

    tweet_list = pd.DataFrame(tweet_list)
    neutral_list = pd.DataFrame(neutral_list)
    negative_list = pd.DataFrame(negative_list)
    positive_list = pd.DataFrame(positive_list)

    print("Total Number: ", len(tweet_list))
    print("No. of Positive Tweets: ", len(positive_list))
    print("No. of neutral Tweets: ", len(neutral_list))
    print("No. of Negative Tweets: ", len(negative_list))
