from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def analyze_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
    output = model(**encoded_input)
    scores = F.softmax(output.logits, dim=1)
    scores = scores.detach().numpy()[0]
    sentiment_score = float(scores[2] - scores[0])  # positive - negative
    return sentiment_score
