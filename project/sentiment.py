# GUPTA211HACKS/project/sentiment.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

labels = ['negative', 'neutral', 'positive']

def analyze_sentiment_batch(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    scores = F.softmax(output.logits, dim=1).numpy()
    results = []
    for i, score in enumerate(scores):
        sentiment_score = float(score[2] - score[0])  # positive - negative
        sentiment = labels[score.argmax()]
        results.append((sentiment, sentiment_score))
    return results
