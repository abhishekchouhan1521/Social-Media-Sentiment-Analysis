# Social-Media-Sentiment-Analysis
# This project uses **Natural Language Processing (NLP)** techniques to perform sentiment analysis on social media data, such as tweets. The goal is to classify tweets as positive, negative, or neutral based on the content of the text. The project uses the **TextBlob** and **scikit-learn** libraries for sentiment analysis and machine learning.

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset (tweets)
df = pd.read_csv('../data/tweets.csv')

# Data Preprocessing: Clean the text data
import re

def clean_text(text):
    # Remove URLs, mentions, special characters
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[\w]*', '', text)  # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower().strip()  # Convert to lowercase and remove extra spaces
    return text

df['cleaned_tweet'] = df['tweet'].apply(clean_text)

# Sentiment Analysis using TextBlob (initial sentiment extraction)
def get_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    return 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'

df['predicted_sentiment'] = df['cleaned_tweet'].apply(get_sentiment)

# Display distribution of sentiments
sns.countplot(x='predicted_sentiment', data=df)
plt.title("Sentiment Distribution")
plt.show()

# Feature Extraction: Convert text to numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_tweet']).toarray()
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model for sentiment classification
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model for later use
joblib.dump(model, '../sentiment_model.pkl')

import tweepy
import pandas as pd

# Set up your Twitter API credentials here
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authenticate to Twitter API
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Function to collect tweets
def collect_tweets(query, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en").items(count)
    tweet_data = []
    
    for tweet in tweets:
        tweet_data.append({
            'tweet': tweet.text,
            'created_at': tweet.created_at
        })
    
    return pd.DataFrame(tweet_data)

# Example usage: Collect 100 tweets about "Python"
df = collect_tweets('Python', count=100)
df.to_csv('tweets.csv', index=False)
import tweepy
import pandas as pd

# Set up your Twitter API credentials here
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authenticate to Twitter API
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Function to collect tweets
def collect_tweets(query, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en").items(count)
    tweet_data = []
    
    for tweet in tweets:
        tweet_data.append({
            'tweet': tweet.text,
            'created_at': tweet.created_at
        })
    
    return pd.DataFrame(tweet_data)

# Example usage: Collect 100 tweets about "Python"
df = collect_tweets('Python', count=100)
df.to_csv('tweets.csv', index=False)

import re

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[\w]*', '', text)  # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower().strip()  # Convert to lowercase and remove extra spaces
    return text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score, classification_report

def perform_sentiment_analysis(df):
    # Feature Extraction using TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_tweet']).toarray()
    y = df['sentiment']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, 'sentiment_model.pkl')

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def evaluate_model(y_test, y_pred):
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
    plt.title("Confusion Matrix")
    plt.show()

def load_model():
    # Load the saved model
    model = joblib.load('sentiment_model.pkl')
    return model
