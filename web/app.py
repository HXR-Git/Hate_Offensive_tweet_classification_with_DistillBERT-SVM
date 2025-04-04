# /Users/harshareddy/Documents/Major_project/MJ/Web/app.py
import os
import re
import torch
import pickle
from flask import Flask, request, render_template # type: ignore

app = Flask(__name__)

# Configure Flask to look for templates in the same directory as app.py
app.template_folder = '.'  # Look for index.html in the Web folder

# Set up paths relative to the project root
base_dir = os.path.dirname(os.path.dirname(__file__))  # Move up one level to MJ/
models_dir = os.path.join(base_dir, 'models')

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the DistilBERT tokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
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

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.lower()
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

# Function to extract embeddings
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
    processed_tweet = preprocess_text(tweet)
    encoding = encode_text(processed_tweet)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    embedding = get_embedding(distilbert_model, input_ids, attention_mask)
    prediction = svm_model.predict(embedding)[0]
    class_labels = {0: "Hate", 1: "Offensive", 2: "Neither"}
    return class_labels[prediction]

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    if not tweet:
        return render_template('index.html', prediction="Please enter a tweet.")
    try:
        prediction = predict_tweet(tweet)
        return render_template('index.html', prediction=f"Predicted Class: {prediction}", tweet=tweet)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)