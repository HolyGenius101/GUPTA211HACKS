import snscrape.modules.twitter as sntwitter
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import streamlit as st
import datetime

st.set_page_config(page_title="Smart Roadmap for Social Movements", layout="centered")
st.title("📊 Smart Roadmap for Social Movements")
st.subheader("Analyze emotional turning points in social conversations")

topic = st.text_input("Enter a social topic (e.g. climate change, AI):", "climate change")
start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))
end_date = st.date_input("End Date", datetime.date(2025, 4, 1))
tweet_limit = st.slider("Number of tweets to analyze", 50, 500, 100, 50)

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
        analyzer = SentimentIntensityAnalyzer()
        df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        weekly = df['sentiment'].resample('W').mean()
        st.markdown("### 📈 Sentiment Timeline")
        st.line_chart(weekly)

        shifts = weekly.diff().abs().sort_values(ascending=False).head(3)
        st.markdown("### 🔀 Top Emotional Turning Points")
        for date, change in shifts.items():
            st.write(f"**{date.date()}** — Change: `{change:.3f}` sentiment units")

        st.markdown("### 🗣️ Sample Tweets")
        st.dataframe(df[['text', 'sentiment']].sample(5))
