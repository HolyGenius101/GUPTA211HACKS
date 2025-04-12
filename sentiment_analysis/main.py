from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
import csv
import urllib.request

app = Flask(__name__)
CORS(app)

# Load Model + Tokenizer
task = "sentiment"
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Load label mapping
labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode("utf-8").split("\n")
    reader = csv.reader(html, delimiter="\t")
    labels = [row[1] for row in reader if len(row) > 1]

# Preprocessing function
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    try:
        data = request.get_json()
        input_text = data.get("text", "")

        if not input_text.strip():
            return jsonify({"error": "No text provided"}), 400

        text = preprocess(input_text)
        encoded_input = tokenizer(text, return_tensors="pt")
        output = model(**encoded_input)
        scores = softmax(output.logits[0].detach().numpy())

        ranking = np.argsort(scores)[::-1]
        top_label = labels[ranking[0]]
        top_score = round(float(scores[ranking[0]]), 4)

        return jsonify({
            "sentiment": top_label,
            "score": top_score
        })

    except Exception as e:
        return jsonify({"error": "Error analyzing sentiment", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
