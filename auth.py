import os
from dotenv import load_dotenv

import tweepy


load_dotenv()

consumerKey = os.getenv("CONSUMER_KEY") 
consumerSecret = os.getenv("CONSUMER_SECRET")
accessToken = os.getenv("ACCESS_TOKEN")
accessTokenSecret = os.getenv("ACCESS_TOKEN_SECRET")

def initialize():
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)

    api = tweepy.API(auth)
    return api
