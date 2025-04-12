import snscrape.modules.twitter as sntwitter
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import datetime

# Streamlit Page Setup
st.set_page_config(page_title="Smart Roadmap for Social Movements", layout="centered")
st.title("📊 Smart Roadmap for Social Movements")
st.subheader("Analyze emotional turning points in social conversations")

# User Input
topic = st.text_input("Enter a social topic (e.g. climate change, AI):", "climate change")
start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))
end_date = st.date_input("End Date", datetime.date(2025, 4, 1))
tweet_limit = st.slider("Number of tweets to analyze", 50, 500, 100, 50)

# Load Hugging Face Model + Tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

labels = ['Negative', 'Neutral', 'Positive']

# Preprocessing function recommended by CardiffNLP
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Function to get sentiment score from Hugging Face
def analyze_sentiment(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    # Custom scoring: positive - negative
    sentiment_score = float(scores[2]) - float(scores[0])
    return sentiment_score

# Main button to analyze
if st.button("🚀 Analyze Now"):
    query = f"{topic} since:{start_date} until:{end_date}"
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= tweet_limit:
            break
        tweets.append([tweet.date, tweet.content])

    df = pd.DataFrame(tweets, columns=['date', 'text'])

    if df.empty:
        st.warning("No tweets found. Try a different topic or date range.")
    else:
        with st.spinner("Analyzing sentiment with RoBERTa..."):
            df['sentiment'] = df['text'].apply(analyze_sentiment)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Resample to weekly sentiment trend
            weekly = df['sentiment'].resample('W').mean()
            st.markdown("### 📈 Sentiment Timeline")
            st.line_chart(weekly)

            # Show biggest changes in sentiment (emotional turning points)
            shifts = weekly.diff().abs().sort_values(ascending=False).head(3)
            st.markdown("### 🔀 Top Emotional Turning Points")
            for date, change in shifts.items():
                st.write(f"**{date.date()}** — Change: `{change:.3f}` sentiment units")

            # Show random sample tweets
            st.markdown("### 🗣️ Sample Tweets")
            st.dataframe(df[['text', 'sentiment']].sample(5))
