from PIL import Image, ImageChops

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
#wordcloud_util.generate_wordcloud(trend_phrases, tweet_frequency, twitter_mask)


""" POS TAGGING """

# Get POS tags for the trend names
#tags = nlp.get_tags(trend_phrases)
# Plot POS tags in descending log scale
#nlp.plot_tags_freq(tags)


""" USING TWEEPY """
api = auth.initialize()

sorted_trends = wordcloud_util.get_sorted_phrasecounts(
    trend_phrases, 
    tweet_frequency
)

""" TEST """
num_tweets = 2
query = "lebron"

test_tweets = tweepy.Cursor(api.search, q=query).items(num_tweets)
nlp.sentiment_analyzer(test_tweets, num_tweets)
