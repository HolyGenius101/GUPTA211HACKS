import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import datetime
import requests
from bs4 import BeautifulSoup

# Setup model + tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
labels = ['negative', 'neutral', 'positive']

# Sentiment analyzer
def analyze_sentiment_batch(texts):
    encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = model(**encoded)
    scores = F.softmax(output.logits, dim=1).numpy()
    sentiments = [labels[s.argmax()] for s in scores]
    sentiment_scores = [float(s[2] - s[0]) for s in scores]  # pos - neg
    return sentiments, sentiment_scores

# Scrape Google News headlines via Bing proxy
headers = {"User-Agent": "Mozilla/5.0"}
def fetch_bing_headlines(topic, date):
    formatted_date = date.strftime('%Y-%m-%d')
    url = f"https://www.bing.com/news/search?q={topic}+after:{formatted_date}+before:{formatted_date}&FORM=HDRSC6"
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    articles = soup.find_all('a', attrs={'class': 'title'})
    seen = set()
    headlines_links = []
    for a in articles:
        text = a.get_text()
        link = a['href']
        if text not in seen:
            seen.add(text)
            headlines_links.append((text, link))
    return headlines_links[:25]

# Streamlit UI
st.set_page_config(page_title="Smart Roadmap", layout="centered")
st.title("📊 Smart Roadmap for Social Movements")
st.markdown("Analyze emotional turning points in news headlines")

# Inputs
topic = st.text_input("Enter a topic (e.g. climate change, AI):", "climate change")
start = st.date_input("Start Date", datetime.date(2024, 1, 1))
end = st.date_input("End Date", datetime.date(2024, 1, 5))
if st.button("Analyze"):
    with st.spinner("Scraping and analyzing..."):
        current = start
        all_data = []
        all_headlines = []
        while current <= end:
            headlines_links = fetch_bing_headlines(topic, current)
            if headlines_links:
                headlines = [t[0] for t in headlines_links]
                links = [t[1] for t in headlines_links]
                sentiments, scores = analyze_sentiment_batch(headlines)
                avg_score = sum(scores) / len(scores)
                all_data.append({"date": current, "avg_score": avg_score})
                for h, s, l in zip(headlines, sentiments, links):
                    hyperlink = f"[{h}]({l})"
                    all_headlines.append({"date": current, "headline": hyperlink, "sentiment": s})
            current += datetime.timedelta(days=1)

        if not all_data:
            st.warning("No headlines found.")
        else:
            df = pd.DataFrame(all_data)
            st.line_chart(df.set_index("date"))
            avg = df['avg_score'].mean()
            mood = "positive" if avg > 0 else "negative" if avg < 0 else "neutral"
            st.markdown(f"**Overall Sentiment:** {mood.title()} (avg score = {avg:.3f})")

            # Show headlines
            st.markdown("### Headlines and Sentiments")
            dfh = pd.DataFrame(all_headlines)
            st.write(dfh.to_markdown(index=False), unsafe_allow_html=True)
