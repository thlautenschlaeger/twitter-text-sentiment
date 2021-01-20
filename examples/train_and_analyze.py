import json
import pandas as pd
import tweepy
import os
import sys

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(".")
from tweesenti.utils import preprocess_string

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(__file__, "../.."))
    data = pd.read_csv(project_root + '/data/twt_sample.csv')

    features = data.text
    labels = data.sentiment

    # Set keyword for sentiment analysis
    keyword = 'tsla'

    # Set number of negative and positive training data to 50/50
    n_positives = labels.value_counts().loc['positive']
    n_negatives = labels.value_counts().loc['negative']
    fraction = n_negatives / n_positives

    # Negatives come first --> ABC
    selected_labels = labels.sort_values()[:int(n_positives * fraction) + n_negatives]
    # Bring features to equal distribution
    features = features[selected_labels.index].sort_index()
    labels = selected_labels.sort_index()

    # Data cleaning
    processed_features_list = []

    for sentence in features:
        processed_feature = preprocess_string(sentence)
        processed_features_list.append(processed_feature)

    # Use bag of words to transform text to numbers with TD-IDF (Term frequency and Inverse Document frequency)
    vectorizer = TfidfVectorizer(max_features=1800, min_df=3, max_df=0.8, stop_words=stopwords.words('english'))
    processed_features = vectorizer.fit_transform(processed_features_list).toarray()

    # Train & test split
    X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.1, random_state=0)

    # Set up the classifier
    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(X_train, y_train)

    # Use the classifier for predictions
    predictions = text_classifier.predict(X_test)

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

    # Collect newest tweets
    for tweet in tweepy.Cursor(api.search, q='#' + keyword, rpp=100).items():
        processed_feature = preprocess_string(tweet.text)
        processed_features_tmp = processed_features_list + [processed_feature]
        processed_features_tmp = vectorizer.fit_transform(processed_features_tmp).toarray()
        processed_tweet = processed_features_tmp[-1]
        # Perform sentiment analysis
        sentiment = text_classifier.predict(processed_tweet.reshape(1, -1))[0]
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
