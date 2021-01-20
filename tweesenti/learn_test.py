import json

import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import tweepy

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('data/twt_sample.csv')

features = data.text
labels = data.sentiment

# Data cleaning
processed_features_list = []


def preprocess_string(sentence):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(sentence))
    # Remove all single characters
    processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    return processed_feature


for sentence in features:
    processed_feature = preprocess_string(sentence)
    processed_features_list.append(processed_feature)

# Use bag of words to transform text to numbers with TD-IDF (Term frequency and Inverse Document frequency)
vectorizer = TfidfVectorizer(max_features=2500, min_df=3, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features_list).toarray()

# Train & test split
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

# Set up the classifier
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

# Use the classifier for predictions
predictions = text_classifier.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

file = json.load(open('./../key.json', 'rb'))
consumer_key = file['consumer_key']
consumer_secret = file['consumer_secret']
bearer_token = file['bearer_token']
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

api = tweepy.API(auth)
p_count, n_count = 0, 0

for tweet in tweepy.Cursor(api.search, q='#tsla', rpp=100).items():
    processed_feature = preprocess_string(tweet.text)
    processed_features_tmp = processed_features_list + [processed_feature]
    processed_features_tmp = vectorizer.fit_transform(processed_features_tmp).toarray()
    processed_tweet = processed_features_tmp[-1]
    sentiment = text_classifier.predict(processed_tweet.reshape(1, -1))[0]
    # print("TWEET: {}; SENTIMENT: {}".format(tweet.text, sentiment[0]))
    if sentiment == 'positive':
        p_count += 1
    else:
        n_count += 1

    print("Positives: {} | Negatives: {} | Total: {}".format(p_count, n_count, p_count+n_count))
