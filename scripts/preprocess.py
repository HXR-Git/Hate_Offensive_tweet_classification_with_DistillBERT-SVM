# preprocess.py (Updated)
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import nltk

# Set NLTK data directory
nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    return text

def augment_data(texts, labels, target_size, class_label):
    aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.3)
    current_size = len(texts)
    if current_size >= target_size:
        return texts, labels
    aug_texts = []
    aug_labels = []
    for text in tqdm(texts, desc=f"Augmenting Class {class_label}"):
        aug_texts.append(text)
        aug_labels.append(class_label)
        for _ in range((target_size // current_size) - 1):
            aug_text = aug.augment(text)[0]
            aug_texts.append(aug_text)
            aug_labels.append(class_label)
    return aug_texts[:target_size], aug_labels[:target_size]

def preprocess_and_augment_data(target_sizes):
    # Load dataset from project root
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up one level to MJ/
    data_path = os.path.join(base_dir, 'labeled_data.csv')
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {data_path}. Please place 'labeled_data.csv' in the project folder.")
        raise

    # Preprocess text
    data['tweet'] = [preprocess_text(t) for t in tqdm(data['tweet'], desc="Preprocessing Tweets")]

    # Augment classes
    hate_data = data[data['class'] == 0]['tweet'].tolist()
    offensive_data = data[data['class'] == 1]['tweet'].tolist()
    neither_data = data[data['class'] == 2]['tweet'].tolist()

    hate_aug_texts, hate_aug_labels = augment_data(hate_data, [0] * len(hate_data), target_sizes[0], 0)
    offensive_aug_texts, offensive_aug_labels = augment_data(offensive_data, [1] * len(offensive_data), target_sizes[1], 1)
    neither_aug_texts, neither_aug_labels = neither_data, [2] * len(neither_data)

    # Combine augmented data
    texts = hate_aug_texts + offensive_aug_texts + neither_aug_texts
    labels = hate_aug_labels + offensive_aug_labels + neither_aug_labels
    return texts, labels

def vectorize_data(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(texts)
    return X_tfidf, vectorizer

def get_data_splits(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
