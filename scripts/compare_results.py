# compare_results.py (Updated)
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get project root directory
def get_project_root():
    return os.path.dirname(os.path.dirname(__file__))  # Go up one level to MJ/

# Set directories
base_dir = get_project_root()
results_dir = os.path.join(base_dir, 'results')
plots_dir = os.path.join(base_dir, 'plots')

os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# List of result files
result_files = [
    'logistic_regression_performance.csv',
    'naive_bayes_performance.csv',
    'svm_performance.csv',
    'ann_performance.csv',
    'distilbert_svm_classifier_performance.csv'
]

# Combine all results
combined_results = pd.DataFrame()
for file in result_files:
    file_path = os.path.join(results_dir, file)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        combined_results = pd.concat([combined_results, df], ignore_index=True)
    else:
        print(f"Warning: {file} not found in {results_dir}. Skipping...")

# Save combined results
combined_results.to_csv(os.path.join(results_dir, 'combined_performance.csv'), index=False)
print("Combined results saved to 'results/combined_performance.csv'")

# Plot comparison
fig, ax = plt.subplots(figsize=(12, 8))
combined_results.plot(x='Model', kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], edgecolor='black')
ax.set_title('Cross-Validation Performance Comparison Across Models', fontsize=14, pad=15)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1)
ax.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'comparison_bar_plot.png'))
plt.close()

print("Comparison bar plot saved to 'plots/comparison_bar_plot.png'")
print("Script completed successfully.")
