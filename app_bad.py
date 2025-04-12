import streamlit as st
from googlenews import GoogleNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def fetch_news(topic, start, end, num_articles):
    gnews = GoogleNews(lang='en')
    gnews.set_time_range(start, end)
    gnews.search(topic)
    results = gnews.results(sort=True)[:num_articles]
    return results

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def display_results(df):
    st.subheader("📊 Sentiment Breakdown")
    counts = df['Sentiment'].value_counts()
    st.bar_chart(counts)

    st.subheader("📰 Headlines and Sentiment")
    for i, row in df.iterrows():
        st.write(f"• {row['Title']} ({row['Sentiment']})")

# Streamlit UI
st.title("🧠 Analyze emotional turning points in social conversations")
topic = st.text_input("Enter a social topic (e.g. climate change, AI):", "climate change")
start_date = st.date_input("Start Date", datetime(2024, 1, 1))
end_date = st.date_input("End Date", datetime(2025, 4, 1))
num_articles = st.slider("Number of Google News articles to analyze", 10, 100, 30)

if st.button("🚀 Analyze Now"):
    with st.spinner("Scraping news and analyzing sentiment..."):
        articles = fetch_news(topic, start_date.strftime("%m/%d/%Y"), end_date.strftime("%m/%d/%Y"), num_articles)
        if not articles:
            st.warning("No articles found. Try a different topic or date range.")
        else:
            df = pd.DataFrame(articles)
            df['Sentiment'] = df['title'].apply(analyze_sentiment)
            df = df.rename(columns={'title': 'Title'})
            display_results(df)
