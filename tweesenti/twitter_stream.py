import json
import tweepy
from conda.gateways.connection import session
import requests


class SymbolStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print(status.text)


if __name__ == '__main__':
    file = json.load(open('./../key.json', 'rb'))
    consumer_key = file['consumer_key']
    consumer_secret = file['consumer_secret']
    bearer_token = file['bearer_token']
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

    api = tweepy.API(auth)

    for tweet in tweepy.Cursor(api.search, q='#bitcoin', rpp=100).items():
        print(tweet.text)


