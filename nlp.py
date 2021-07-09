import nltk
import matplotlib.pyplot as plt
import pandas as pd
import re

from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import utils


# Setup nltk resources
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
            
    tweet_list = pd.DataFrame(tweet_list)
    analyze_tweets(tweet_list)


def generate_sentiment_chart(query, idx, positive, negative, neutral):
    plt.figure()
    labels = [
        f"Positive [{positive}%]",
        f"Negative [{negative}%]",
        f"Neutral [{neutral}%]",
    ]
    sizes = [positive, negative, neutral]
    colors = ["yellowgreen", "red", "blue"]
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.style.use("default")
    plt.legend(labels)
    plt.title(f"Sentiment Analysis for trend - {query}")
    plt.axis("equal")
    plt.savefig(f"results/sentiment_{idx}.png")


def analyze_tweets(tweet_list):
    tweet_list.drop_duplicates(inplace=True)

    tw_list = pd.DataFrame(tweet_list)
    tw_list["text"] = tw_list[0]

    # Remove punctuation
    remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
    rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([0-9A-Za-z\t])|(\w+:\/\/\S+)"," ",x)
    tw_list["text"] = tw_list.text.map(remove_rt)
    tw_list["text"] = tw_list.text.str.lower()
    
    tw_list[["polarity", "subjectivity"]] = tw_list["text"].apply(
        lambda Text: pd.Series(TextBlob(Text).sentiment)
    )
    for index, row in tw_list["text"].iteritems():
        score = SentimentIntensityAnalyzer().polarity_scores(row)
        neg = score["neg"]
        pos = score["pos"]
        neu = score["neu"]
        comp = score["compound"]
        if neg > pos:
            tw_list.loc[index, "sentiment"] = "negative"
        elif pos > neg:
            tw_list.loc[index, "sentiment"] =  "positive"
        else:
            tw_list.loc[index, "sentiment"] = "neutral"

        tw_list.loc[index, "neg"] = neg
        tw_list.loc[index, "pos"] = pos
        tw_list.loc[index, "neu"] = neu
        tw_list.loc[index, "compound"] = comp

    tw_list.to_csv("results/tweets_with_sentiments.csv")

    tw_list_negative = tw_list[tw_list["sentiment"] == "negative"]
    tw_list_positive = tw_list[tw_list["sentiment"] == "positive"]
    tw_list_neutral = tw_list[tw_list["sentiment"] == "neutral"]

    sentiment_counts = utils.count_values_in_column(tw_list, "sentiment")
    sentiment_counts.to_csv("results/sentiment_counts.csv")

    plt.figure()
    names = sentiment_counts.index
    size=sentiment_counts["Percentage"]

    circle = plt.Circle((0,0), 0.7, color="white")
    plt.pie(size, labels=names, colors=["green", "red", "blue"])
    p = plt.gcf()
    p.gca().add_artist(circle)
    plt.savefig("results/sentiment_2.png")
