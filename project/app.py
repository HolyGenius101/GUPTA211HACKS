import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from sentiment import analyze_sentiment
import random

st.set_page_config(page_title="Sentiment Tracker", page_icon="📈")
st.title("📈 Daily Sentiment Tracker")
st.write("Analyze the emotional trend of a topic over time.")

topic = st.text_input("Enter a topic:", "climate change")
start_date = st.text_input("Start Date (YYYY/MM/DD)", "2024/01/01")
end_date = st.text_input("End Date (YYYY/MM/DD)", "2024/01/10")
articles_per_day = st.slider("Number of headlines per day", 5, 50, 25)

def generate_sample_headline(topic):
    templates = [
        f"{topic} sparks concern",
        f"Debate over {topic}",
        f"{topic} policies in focus",
        f"New study reveals {topic} impact",
        f"Protests erupt around {topic}",
        f"Hope rises with {topic} solutions",
        f"Mixed public opinion on {topic}",
        f"{topic} dominates news cycle",
        f"{topic} trend continues",
        f"Growing awareness about {topic}"
    ]
    return random.choice(templates)

if st.button("🚀 Analyze"):
    try:
        start = datetime.strptime(start_date, "%Y/%m/%d")
        end = datetime.strptime(end_date, "%Y/%m/%d")
    except:
        st.error("Invalid date format. Please use YYYY/MM/DD.")
        st.stop()

    days = (end - start).days + 1
    results = []

    with st.spinner("Analyzing..."):
        for i in range(days):
            date = start + timedelta(days=i)
            scores = [analyze_sentiment(generate_sample_headline(topic)) for _ in range(articles_per_day)]
            avg_score = np.mean(scores)
            results.append((date.date(), avg_score))

    df = pd.DataFrame(results, columns=["Date", "Avg Sentiment Score"])

    st.subheader("📊 Sentiment Trend Over Time")
    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"], df["Avg Sentiment Score"], marker='o')
    plt.xlabel("Date")
    plt.ylabel("Avg Sentiment Score (Pos - Neg)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    st.subheader("🗂️ Data")
    st.dataframe(df)
