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
    plt.savefig(f"results/pos-tag-frequency.png")


def sentiment_analyzer(tweets, num_tweets, query, rank):
    tweet_list = []
    rt_counts = []
    favorite_counts = []

    for tweet in tweets:
        tweet_list.append(tweet.text)
        rt_counts.append(int(tweet.retweet_count))
        favorite_counts.append(int(tweet.favorite_count))
            
    tweet_list = pd.DataFrame(tweet_list)
    return analyze_tweets(tweet_list, rt_counts, favorite_counts, query, rank)


def analyze_tweets(tweet_list, rt_counts, favorite_counts, query, rank):
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
        tw_list.loc[index, "retweets"] = rt_counts[index]
        tw_list.loc[index, "favoriites"] = favorite_counts[index]

    tw_list.to_csv(f"results/{rank}__tweets-with-sentiments.csv")

    tw_list_negative = tw_list[tw_list["sentiment"] == "negative"]
    tw_list_positive = tw_list[tw_list["sentiment"] == "positive"]
    tw_list_neutral = tw_list[tw_list["sentiment"] == "neutral"]

    total_count = len(tw_list)
    negative_count = len(tw_list_negative)
    positive_count = len(tw_list_positive)
    neutral_count = len(tw_list_neutral)

    positive_per = utils.percentage(positive_count, total_count)
    negative_per = utils.percentage(negative_count, total_count)
    neutral_per = utils.percentage(neutral_count, total_count)

    total_rts = sum(rt_counts)
    total_favs = sum(favorite_counts)
    
    sentiment_counts = utils.count_values_in_column(tw_list, "sentiment")
    sentiment_counts["total_retweets"] = total_rts
    sentiment_counts["total_favorites"] = total_favs
    sentiment_counts.to_csv(f"results/{rank}__sentiment-counts.csv")

    plt.figure()
    labels = [
        f"Positive [{str(positive_per)}%]", 
        f"Neutral [{str(neutral_per)}%]",
        f"Negative [{str(negative_per)}%]"
    ]
    sizes = [positive_count, neutral_count, negative_count]

    plt.pie(
        sizes, 
        #labels=labels, 
        colors=["yellowgreen", "blue", "red"],
        startangle=90,
        radius=0.6,
    )
    plt.legend(labels)
    plt.title(f"Sentiment for {query}")
    plt.savefig(f"results/{rank}__sentiment.png")

    return [positive_per, negative_per, neutral_per, total_rts, total_favs]
