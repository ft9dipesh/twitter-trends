from PIL import Image, ImageChops
from langdetect import detect
from itertools import islice

import pandas as pd
import numpy as np
import tweepy

import auth
import wordcloud_util
import nlp


""" READ DATA """

# Contains trends__name, trends__url, trends__promoted_content,
# trends__query, trends__tweet_volume, as_of columns
df = pd.read_csv(r"trending_data__final.csv", encoding="utf-8")


# Get Trend phrases, and their frequencies
trend_phrases = df.iloc[:,0]
tweet_frequency = df.iloc[:,4]

""" WORDCLOUD """

# Get a black and white twitter logo mask for wordcloud
twitter_logo = Image.open("twitter_logo.png")
twitter_mask = np.array(ImageChops.invert(twitter_logo))

# Generate a wordcloud from name, and tweet_volume
wordcloud_util.generate_wordcloud(trend_phrases, tweet_frequency, twitter_mask)


""" POS TAGGING """

# Get POS tags for the trend names
tags = nlp.get_tags(trend_phrases)
# Plot POS tags in descending log scale
nlp.plot_tags_freq(tags)


""" USING TWEEPY """
api = auth.initialize()

sorted_trends = wordcloud_util.get_sorted_phrasecounts(
    trend_phrases, 
    tweet_frequency
)


""" TEST """
num_tweets = 10
num_trends_to_analyze = 1

idx=1

group_results = []
res = pd.DataFrame(group_results)

for query, freq in islice(sorted_trends.items(), 0, num_trends_to_analyze):
    query_tweets = tweepy.Cursor(
        api.search, 
        q=query, 
        lang="en", 
        show_user=False
    ).items(num_tweets)

    [
        positive_per, 
        negative_per, 
        neutral_per, 
        total_rts, 
        total_favs,
    ] = nlp.sentiment_analyzer(query_tweets, num_tweets, query, idx)
    
    res.loc[idx, "Trend"] = query
    res.loc[idx, "Positive %"] = positive_per
    res.loc[idx, "Negative %"] = negative_per
    res.loc[idx, "Neutral %"] = neutral_per
    res.loc[idx, "Total Retweets"] = total_rts
    res.loc[idx, "Total Favorites"] = total_favs

    idx+=1


res.to_csv("results/trend_results.csv")
