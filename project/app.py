import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sentiment import analyze_sentiment
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="Smart Roadmap for Social Movements", page_icon="📊")
st.title("📊 Smart Roadmap for Social Movements")
st.write("Analyze emotional turning points in social conversations")

# Inputs
topic = st.text_input("Enter a social topic (e.g. climate change, AI):", "climate change")
start_date = st.text_input("Start Date (YYYY/MM/DD)", "2024/01/01")
end_date = st.text_input("End Date (YYYY/MM/DD)", "2025/04/01")
num_samples = st.slider("Number of sample headlines to analyze", 10, 100, 20)

if st.button("🚀 Analyze Now"):
    st.info("Simulating data...")

    def generate_sample_headline(topic):
        templates = [
            f"{topic} sparks global concern",
            f"Rising {topic} rates cause debate",
            f"New policies target {topic}",
            f"{topic.capitalize()} movement gains traction",
            f"Experts clash over {topic} strategy",
            f"Protests erupt over {topic}",
            f"Tech solutions proposed for {topic}",
            f"Social media reacts to {topic}",
            f"{topic.capitalize()} dominates headlines",
            f"Latest updates on {topic}"
        ]
        return random.choice(templates)

    # Date range handling
    try:
        start_dt = datetime.strptime(start_date, "%Y/%m/%d")
        end_dt = datetime.strptime(end_date, "%Y/%m/%d")
    except:
        st.error("Invalid date format. Use YYYY/MM/DD.")
        st.stop()

    # Create fake headlines with spread dates
    headlines = []
    dates = []
    for i in range(num_samples):
        headlines.append(generate_sample_headline(topic))
        date = start_dt + (end_dt - start_dt) * i / max(num_samples - 1, 1)
        dates.append(date.date())

    # Analyze sentiments
    sentiment_labels = []
    score_labels = []
    scores = []

    for headline in headlines:
        sentiment, score = analyze_sentiment(headline)
        scores.append(score)

        label = {
            "positive": "😊 Positive",
            "neutral": "😐 Neutral",
            "negative": "😠 Negative"
        }[sentiment]
        sentiment_labels.append(label)

        if score > 0.5:
            score_desc = f"Strong Positive ({score:+.2f})"
        elif score > 0.1:
            score_desc = f"Mild Positive ({score:+.2f})"
        elif score < -0.5:
            score_desc = f"Strong Negative ({score:+.2f})"
        elif score < -0.1:
            score_desc = f"Mild Negative ({score:+.2f})"
        else:
            score_desc = f"Neutral ({score:+.2f})"
        score_labels.append(score_desc)

    # Build DataFrame
    df = pd.DataFrame({
        "Date": dates,
        "Headline": headlines,
        "Sentiment": sentiment_labels,
        "Score": score_labels,
        "Raw Score": scores
    })
    
    # Plot sentiment trend over time
    st.subheader("📉 Sentiment Trend Over Time")
    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"], df["Raw Score"], marker='o')
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score (positive - negative)")
    plt.tight_layout()
    st.pyplot(plt)

    # Display data table
    st.subheader("📰 Headlines and Sentiment")
    st.dataframe(df.drop(columns="Raw Score").reset_index(drop=True))
