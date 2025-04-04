# Tweet Classification Project

This project classifies English tweets into three categories: **Hate**, **Offensive**, and **Neither**, using various machine learning models. The project uses a dataset of labeled English tweets and employs DistilBERT embeddings and traditional ML models for classification.

## Project Structure
- **data/**: Contains the dataset (`labeled_data.csv`).
- **models/**: Contains trained models (e.g., `distilbert_model.pth`, `svm_model.pkl`).
- **results/**: Contains performance metrics (e.g., `combined_performance.csv`).
- **plots/**: Contains visualizations (e.g., `comparison_bar_plot.png`).
- **scripts/**: Contains all training and inference scripts.
- **nltk_data/**: Contains NLTK resources for augmentation.

## Dataset
- **File**: `data/labeled_data.csv`
- **Size**: 24,783 original rows, augmented to 37,653 rows during training.
- **Classes**: 
  - 0: Hate
  - 1: Offensive
  - 2: Neither

## Models and Performance
- **DistilBERT-SVM**: Accuracy 0.94, Precision 0.94, Recall 0.94, F1-Score 0.94
- **Logistic Regression**: Mean Accuracy 0.91
- **Naive Bayes**: Mean Accuracy 0.85
- **SVM (TF-IDF)**: Mean Accuracy 0.92
- **ANN**: Mean Accuracy 0.91
- **Results**: Saved in `results/` (e.g., `combined_performance.csv`).
- **Plots**: Saved in `plots/` (e.g., `comparison_bar_plot.png`).

## Scripts
- **distilled_acc.py**: Trains DistilBERT-SVM model.
- **logistic_regression.py**: Trains Logistic Regression with TF-IDF.
- **naive_bayes.py**: Trains Naive Bayes with TF-IDF.
- **svm.py**: Trains SVM with TF-IDF.
- **ann.py**: Trains ANN with TF-IDF.
- **compare_results.py**: Combines results and generates a comparison plot.
- **predict_tweet.py**: Performs real-time inference on new tweets.

## Usage
1. Activate the virtual environment:
   ```bash
   source tf_env/bin/activate
