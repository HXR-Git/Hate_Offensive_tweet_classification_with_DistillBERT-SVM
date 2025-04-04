# distilled_acc.py (Single Split, Targeting >95% Accuracy, ~4-5 Hours on MPS)
print("Starting script...")

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import nlpaug.augmenter.word as naw
import nltk
from tqdm import tqdm
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Create output directories
base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up one level to MJ/
results_dir = os.path.join(base_dir, 'results')
plots_dir = os.path.join(base_dir, 'plots')
models_dir = os.path.join(base_dir, 'models')
confusion_matrices_dir = os.path.join(plots_dir, 'confusion_matrices')

os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(confusion_matrices_dir, exist_ok=True)

# Set NLTK data directory
nltk_data_dir = os.path.join(base_dir, 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)
print("NLTK downloads complete.")

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
data_path = os.path.join(base_dir, 'labeled_data.csv')
try:
    data = pd.read_csv(data_path)
    print("Dataset loaded successfully.")
    print("Dataset Preview:")
    print(data[['tweet', 'class']].head())
except FileNotFoundError:
    print(f"Error: Dataset file not found at {data_path}.")
    raise

# Helper function for encoding texts
def encode_texts(texts, max_length=128):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    return text

data['tweet'] = [preprocess_text(t) for t in tqdm(data['tweet'], desc="Preprocessing Tweets")]

# Class distribution before augmentation
print("\nClass Distribution Before Augmentation:")
print(data['class'].value_counts(normalize=True) * 100)

# EDA Augmentation
aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.5)  # Increased to 0.5 for more diversity

def augment_data(texts, labels, target_size, class_label):
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

# Augment classes
hate_data = data[data['class'] == 0]['tweet'].tolist()
offensive_data = data[data['class'] == 1]['tweet'].tolist()
neither_data = data[data['class'] == 2]['tweet'].tolist()

hate_aug_texts, hate_aug_labels = augment_data(hate_data, [0] * len(hate_data), 15000, 0)
offensive_aug_texts, offensive_aug_labels = augment_data(offensive_data, [1] * len(offensive_data), 20000, 1)
neither_aug_texts, neither_aug_labels = neither_data, [2] * len(neither_data)

# Combine augmented data
augmented_data = pd.DataFrame({
    'tweet': hate_aug_texts + offensive_aug_texts + neither_aug_texts,
    'class': hate_aug_labels + offensive_aug_labels + neither_aug_labels
})

print("\nClass Distribution After Augmentation:")
print(augmented_data['class'].value_counts(normalize=True) * 100)
print(f"New dataset size: {len(augmented_data)}")

# Single train-test split
X = augmented_data['tweet'].values
y = augmented_data['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize and encode data
train_encodings = encode_texts(X_train)
test_encodings = encode_texts(X_test)

train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    torch.tensor(y_train)
)
test_dataset = TensorDataset(
    test_encodings['input_ids'],
    test_encodings['attention_mask'],
    torch.tensor(y_test)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.to(device)

# Training function with learning rate scheduler
def train_model(model, train_loader, epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

# Train DistilBERT
print("Starting DistilBERT Training...")
train_model(model, train_loader, epochs=5)

# Extract embeddings function
def get_embeddings(model, loader):
    model.eval()
    embeddings = []
    labels_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Embeddings"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model.distilbert(input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            labels_list.append(labels.cpu().numpy())
    return np.vstack(embeddings), np.hstack(labels_list)

# Get embeddings
train_embeddings, train_labels = get_embeddings(model, train_loader)
test_embeddings, test_labels = get_embeddings(model, test_loader)

# Train SVM on embeddings
print("Starting SVM Training...")
svm = SVC(kernel='linear', probability=True, random_state=42, C=10)
svm.fit(train_embeddings, train_labels)

# Save the trained SVM model
with open(os.path.join(models_dir, 'svm_model.pkl'), 'wb') as f:
    pickle.dump(svm, f)
print("Saved SVM model to 'models/svm_model.pkl'")

# Save the trained DistilBERT model
torch.save(model.state_dict(), os.path.join(models_dir, 'distilbert_model.pth'))
print("Saved DistilBERT model to 'models/distilbert_model.pth'")

# Evaluate ensemble
y_pred = svm.predict(test_embeddings)
accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred, average='weighted')
recall = recall_score(test_labels, y_pred, average='weighted')
f1 = f1_score(test_labels, y_pred, average='weighted')

print("\nDistilBERT-SVM Classifier Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Confusion Matrix
cm = confusion_matrix(test_labels, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Hate', 'Offensive', 'Neither'], yticklabels=['Hate', 'Offensive', 'Neither'])
plt.title('Confusion Matrix - DistilBERT-SVM Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(confusion_matrices_dir, 'confusion_matrix_distilbert_svm.png'))
plt.close()

# Save results
results = pd.DataFrame({
    'Model': ['DistilBERT-SVM Classifier'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-Score': [f1]
})

results.to_csv(os.path.join(results_dir, 'distilbert_svm_classifier_performance.csv'), index=False)
print("\nResults saved to 'results/distilbert_svm_classifier_performance.csv'")

# Bar Plot
fig, ax = plt.subplots(figsize=(10, 6))
results.plot(x='Model', kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], edgecolor='black')
ax.set_title('Performance of DistilBERT-SVM Classifier', fontsize=14, pad=15)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1)
ax.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=0, fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'distilbert_svm_classifier_performance_bar_plot.png'))
plt.close()

print("Script completed successfully.")