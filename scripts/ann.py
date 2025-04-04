# ann.py (Updated)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
import pickle
from preprocess import preprocess_and_augment_data, vectorize_data, get_data_splits

# Get project root directory
def get_project_root():
    return os.path.dirname(os.path.dirname(__file__))  # Go up one level to MJ/

# Set output directories
base_dir = get_project_root()
results_dir = os.path.join(base_dir, 'results')
plots_dir = os.path.join(base_dir, 'plots')
confusion_matrices_dir = os.path.join(plots_dir, 'confusion_matrices')
models_dir = os.path.join(base_dir, 'models')

os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(confusion_matrices_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load and preprocess data with augmentation
texts, labels = preprocess_and_augment_data(target_sizes={0: 15000, 1: 20000, 2: 10000})

# Vectorize data
X_tfidf, vectorizer = vectorize_data(texts)

# Cross-validation setup
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_tfidf)):
    print(f"Fold {fold + 1}/5")
    X_train_fold, X_val_fold = X_tfidf[train_idx], X_tfidf[val_idx]
    y_train_fold, y_val_fold = [labels[i] for i in train_idx], [labels[i] for i in val_idx]

    # Train ANN (MLP)
    ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    ann.fit(X_train_fold, y_train_fold)

    # Evaluate
    y_pred_fold = ann.predict(X_val_fold)
    metrics['accuracy'].append(accuracy_score(y_val_fold, y_pred_fold))
    metrics['precision'].append(precision_score(y_val_fold, y_pred_fold, average='weighted'))
    metrics['recall'].append(recall_score(y_val_fold, y_pred_fold, average='weighted'))
    metrics['f1'].append(f1_score(y_val_fold, y_pred_fold, average='weighted'))

# Compute mean and standard deviation of metrics
print("\nCross-Validation Results for ANN:")
print(f"Mean Accuracy: {np.mean(metrics['accuracy']):.2f} (+/- {np.std(metrics['accuracy']):.2f})")
print(f"Mean Precision: {np.mean(metrics['precision']):.2f} (+/- {np.std(metrics['precision']):.2f})")
print(f"Mean Recall: {np.mean(metrics['recall']):.2f} (+/- {np.std(metrics['recall']):.2f})")
print(f"Mean F1-Score: {np.mean(metrics['f1']):.2f} (+/- {np.std(metrics['f1']):.2f})")

# Final train-test split for confusion matrix
X_train, X_test, y_train, y_test = get_data_splits(X_tfidf.toarray(), labels, test_size=0.2, random_state=42)
ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
ann.fit(X_train, y_train)

# Save the trained ANN model
with open(os.path.join(models_dir, 'ann_model.pkl'), 'wb') as f:
    pickle.dump(ann, f)
print("Saved ANN model to 'models/ann_model.pkl'")

# Evaluate on test set
y_pred = ann.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Hate', 'Offensive', 'Neither'], yticklabels=['Hate', 'Offensive', 'Neither'])
plt.title('Confusion Matrix - ANN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(confusion_matrices_dir, 'confusion_matrix_ann_full.png'))
plt.close()

# Save cross-validation results
results = pd.DataFrame({
    'Model': ['ANN'],
    'Accuracy': [np.mean(metrics['accuracy'])],
    'Precision': [np.mean(metrics['precision'])],
    'Recall': [np.mean(metrics['recall'])],
    'F1-Score': [np.mean(metrics['f1'])]
})
results.to_csv(os.path.join(results_dir, 'ann_performance.csv'), index=False)
print("Results saved to 'results/ann_performance.csv'")

# Bar Plot for cross-validation results
fig, ax = plt.subplots(figsize=(10, 6))
results.plot(x='Model', kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], edgecolor='black')
ax.set_title('Cross-Validation Performance of ANN', fontsize=14, pad=15)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1)
ax.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=0, fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'ann_performance_bar_plot_cv.png'))
plt.close()

print("Script completed successfully.")
