import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sentiment import analyze_sentiment
import datetime

st.title("📊 Smart Roadmap for Social Movements")
st.write("Analyze emotional turning points in social conversations")

topic = st.text_input("Enter a social topic (e.g. climate change, AI):", "climate change")
start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))
end_date = st.date_input("End Date", datetime.date(2025, 4, 1))
num_items = st.slider("Number of sample headlines to analyze", 10, 100, 30)

if st.button("🚀 Analyze Now"):
    st.info("Using sample data to simulate results...")

    headlines = [f"{topic} headline {i}" for i in range(num_items)]
    sentiments, scores = [], []

    for text in headlines:
        sent, score = analyze_sentiment(text)
        sentiments.append(sent)
        scores.append(score)

    df = pd.DataFrame({
        "Headline": headlines,
        "Sentiment": sentiments,
        "Score": scores
    })

    st.subheader("📈 Sentiment Trend")
    st.line_chart(df["Score"])

    st.subheader("🗞️ Headlines and Sentiment")
    st.dataframe(df)
