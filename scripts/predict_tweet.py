# predict_tweet.py (Real-Time Tweet Classification)
import os
import re
import torch
import pickle
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up paths
base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up one level to MJ/
models_dir = os.path.join(base_dir, 'models')

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the trained DistilBERT model
distilbert_model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=3
)
distilbert_model.load_state_dict(
    torch.load(os.path.join(models_dir, '/Users/harshareddy/Documents/Major_project/MJ/models/distilbert_model.pth'), map_location=device)
)
distilbert_model.to(device)
distilbert_model.eval()
print("Loaded DistilBERT model.")

# Load the trained SVM model
with open(os.path.join(models_dir, '/Users/harshareddy/Documents/Major_project/MJ/models/svm_model.pkl'), 'rb') as f:
    svm_model = pickle.load(f)
print("Loaded SVM model.")

# Preprocessing function (same as in distilled_acc.py)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    return text

# Function to encode a single tweet
def encode_text(text, max_length=128):
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    return encoding

# Function to extract embeddings from a single tweet
def get_embedding(model, input_ids, attention_mask):
    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model.distilbert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embedding

# Function to predict the class of a tweet
def predict_tweet(tweet):
    # Preprocess the tweet
    processed_tweet = preprocess_text(tweet)
    print(f"Processed Tweet: {processed_tweet}")

    # Encode the tweet
    encoding = encode_text(processed_tweet)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Extract embedding
    embedding = get_embedding(distilbert_model, input_ids, attention_mask)

    # Predict with SVM
    prediction = svm_model.predict(embedding)[0]

    # Map prediction to class label
    class_labels = {0: "Hate", 1: "Offensive", 2: "Neither"}
    predicted_class = class_labels[prediction]

    return predicted_class

# Main loop for real-time prediction
def main():
    print("Real-Time Tweet Classification")
    print("Enter a tweet to classify (or type 'exit' to quit):")
    
    while True:
        # Get user input
        tweet = input("Tweet: ").strip()
        
        # Check for exit condition
        if tweet.lower() == 'exit':
            print("Exiting...")
            break
        
        if not tweet:
            print("Please enter a non-empty tweet.")
            continue
        
        # Predict the class
        try:
            predicted_class = predict_tweet(tweet)
            print(f"Predicted Class: {predicted_class}\n")
        except Exception as e:
            print(f"Error during prediction: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()