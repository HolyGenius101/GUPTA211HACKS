from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

labels = ['negative', 'neutral', 'positive']

def analyze_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
    output = model(**encoded_input)
    scores = F.softmax(output.logits, dim=1)
    scores = scores.detach().numpy()[0]
    sentiment_score = float(scores[2] - scores[0])  # pos - neg
    sentiment = labels[scores.argmax()]
    return sentiment, sentiment_score
