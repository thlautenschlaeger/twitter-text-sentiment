import json
import time

import pandas as pd
import tweepy
import os
import sys
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(".")
from tweesenti.utils import preprocess_string

if __name__ == '__main__':
    # Set keyword for sentiment analysis
    keyword = '$DOGE'
    # Sentiment summarized in minute bars
    time_interval_minutes = 1
    project_root = os.path.abspath(os.path.join(__file__, "../.."))

    # Load training data
    sentiment_data = pd.read_csv(project_root + '/data/twt_sample.csv')
    spam_data = pd.read_csv(project_root + '/data/twt_spam.csv')

    # Extract features and labels
    sentiment_features = sentiment_data.text
    sentiment_labels = sentiment_data.sentiment

    spam_features = spam_data.Tweet
    spam_labels = spam_data.Type

    # Set number of negative and positive training data to 50/50
    n_positives = sentiment_labels.value_counts().loc['positive']
    n_negatives = sentiment_labels.value_counts().loc['negative']
    fraction = n_negatives / n_positives

    # Negatives come first --> ABC
    selected_labels = sentiment_labels.sort_values()[:int(n_positives * fraction) + n_negatives]
    # Bring features to equal distribution
    sentiment_features = sentiment_features[selected_labels.index].sort_index()
    sentiment_labels = selected_labels.sort_index()

    # Data cleaning
    processed_sentiment_features_list = []
    for sentence in sentiment_features:
        processed_sentiment_feature = preprocess_string(sentence)
        processed_sentiment_features_list.append(processed_sentiment_feature)

    # Use bag of words to transform text to numbers with TD-IDF (Term frequency and Inverse Document frequency)
    sentiment_vectorizer = TfidfVectorizer(max_features=1800, min_df=3, max_df=0.8,
                                           stop_words=stopwords.words('english'))
    processed_sentiment_features = sentiment_vectorizer.fit_transform(processed_sentiment_features_list).toarray()

    # Train & test split
    X_train, X_test, y_train, y_test = train_test_split(processed_sentiment_features, sentiment_labels, test_size=0.1,
                                                        random_state=0)

    # Set up the classifier
    sentiment_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    sentiment_classifier.fit(X_train, y_train)

    # Use the classifier for predictions
    predictions = sentiment_classifier.predict(X_test)

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))

    # Train spam detection
    processed_spam_features_list = []
    for sentence in spam_features:
        processed_spam_feature = preprocess_string(sentence)
        processed_spam_features_list.append(processed_spam_feature)

    # Use bag of words to transform text to numbers with TD-IDF (Term frequency and Inverse Document frequency)
    spam_vectorizer = TfidfVectorizer(max_features=1800, min_df=3, max_df=0.8, stop_words=stopwords.words('english'))
    processed_spam_features = spam_vectorizer.fit_transform(processed_spam_features_list).toarray()

    # Train & test split
    X_train, X_test, y_train, y_test = train_test_split(processed_spam_features, spam_labels, test_size=0.1,
                                                        random_state=0)

    # Set up the classifier
    spam_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    spam_classifier.fit(X_train, y_train)

    # Use the classifier for predictions
    predictions = spam_classifier.predict(X_test)

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))

    # Authenticate with twitter api
    file = json.load(open(project_root + '/key.json', 'rb'))
    consumer_key = file['consumer_key']
    consumer_secret = file['consumer_secret']
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    api = tweepy.API(auth)

    # Initialize positive and negative counts
    p_count, n_count = 0, 0

    time_bars = pd.DataFrame()

    while True:
        # Synchronize clock to fill bars
        sleep_time = 60 - (datetime.now() - timedelta(seconds=60)).second
        time.sleep(sleep_time)

        start = time.perf_counter()

        # Collect newest tweets
        for tweet in tweepy.Cursor(api.search, q='#' + keyword, rpp=100).items():
            processed_feature = preprocess_string(tweet.text)

            processed_features_tmp = processed_spam_features_list + [processed_feature]
            processed_features_tmp = spam_vectorizer.fit_transform(processed_features_tmp).toarray()
            processed_tweet = processed_features_tmp[-1]

            # Perform spam detection
            spam = spam_classifier.predict(processed_tweet.reshape(1, -1))[0]

            if spam != 'Spam':
                processed_features_tmp = processed_sentiment_features_list + [processed_feature]
                processed_features_tmp = sentiment_vectorizer.fit_transform(processed_features_tmp).toarray()
                processed_tweet = processed_features_tmp[-1]
                # Perform sentiment analysis
                sentiment = sentiment_classifier.predict(processed_tweet.reshape(1, -1))[0]
                # print("TWEET: {}; SENTIMENT: {}".format(tweet.text, sentiment[0]))
                if sentiment == 'positive':
                    p_count += 1
                else:
                    n_count += 1

                totals = p_count + n_count
                print("[Num positives: {} - weight: {:.2f}%]| [Num negatives: {} - "
                      "weight: {:.2f}%] | Total: {}".format(p_count,
                                                            p_count / totals * 100,
                                                            n_count,
                                                            n_count / totals * 100,
                                                            totals))
            # if (datetime.now() - timedelta(seconds=60)).second <= 0.1 and p_count + n_count > 0:
            if time.perf_counter() - start >= 60. and p_count + n_count > 0:
                s_score = (2 * (p_count / (p_count + n_count))) - 1
                time_bars = time_bars.append(pd.DataFrame({'n_positives': p_count,
                                                           'n_negatives': n_count,
                                                           'sentiment_score': s_score},
                                                          index=[pd.Timestamp.utcnow()]))

                p_count = 0
                n_count = 0

                ax = time_bars['sentiment_score'].plot()
                ax.set_xlabel('Time')
                ax.set_ylabel('Sentiment')
                ax.set_ylim(ymin=-1, ymax=1)
                ax.axhline(y=0, ls='--', color='black')
                ax.set_title(keyword + ' sentiment')
                plt.savefig('sentiment_' + keyword + '.png')
                # plt.show()
                plt.close('all')

                start = time.perf_counter()
