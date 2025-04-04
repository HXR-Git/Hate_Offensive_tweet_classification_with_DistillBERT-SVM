import os
import subprocess
from graphviz import Digraph

# Debug: Print the PATH environment variable
print("Current PATH:", os.environ.get('PATH'))

# Ensure Graphviz is in the PATH
# Add common Graphviz paths (adjust based on your system)
graphviz_paths = [
    '/opt/homebrew/bin',  # Apple Silicon (Homebrew)
    '/usr/local/bin',     # Intel (Homebrew)
    '/usr/bin',           # System default
]
current_path = os.environ.get('PATH', '')
new_path = current_path
for path in graphviz_paths:
    if path not in current_path:
        new_path = f"{path}:{new_path}"
os.environ['PATH'] = new_path

# Debug: Verify dot is accessible
try:
    dot_version = subprocess.run(['dot', '-V'], capture_output=True, text=True)
    print("dot version:", dot_version.stderr.strip())
except FileNotFoundError:
    print("dot executable not found in PATH. Please ensure Graphviz is installed correctly.")

# Set base directory (MJ/)
base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script (MJ/)

# Create a new directed graph
dot = Digraph(comment='Workflow Diagram')

# Add nodes (stages)
dot.node('A', 'Dataset Acquisition: labeled_data.csv (24,783 tweets)')
dot.node('B', 'Data Preprocessing')
dot.node('C', 'Model Training')
dot.node('D', 'Fine-tune DistilBERT')
dot.node('E', 'Train Baseline Models')
dot.node('F', 'Train SVM Classifier')
dot.node('G', 'Evaluate Baselines')
dot.node('H', 'Real-Time Inference System (Flask)')
dot.node('I', 'End')

# Add edges
dot.edges(['AB', 'BC', 'CD', 'CE', 'DF', 'EG', 'FH', 'HI'])

# Add subgraphs for parallel processes
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('D')
    s.node('E')

# Add labels for sub-steps (as annotations)
dot.node('B1', ' - Convert to lowercase\n - Remove special characters, URLs, mentions\n - Remove stop words (NLTK)\n - Lemmatization (WordNet)\n - Data augmentation (+20% Hate/Neither)\n - Tokenization (DistilBERT, max 128)', shape='box')
dot.edge('B', 'B1', style='dashed')
dot.node('D1', ' - 6-layer architecture\n - [CLS] embedding (768D)\n - AdamW, 3 epochs', shape='box')
dot.edge('D', 'D1', style='dashed')
dot.node('E1', ' - Logistic Regression (TF-IDF)\n - Naive Bayes (TF-IDF)\n - SVM (TF-IDF)\n - ANN (DistilBERT)', shape='box')
dot.edge('E', 'E1', style='dashed')
dot.node('F1', ' - RBF kernel\n - C = 1.0, gamma = scale', shape='box')
dot.edge('F', 'F1', style='dashed')
dot.node('H1', ' - Preprocess tweet\n - Extract embeddings\n - Classify (SVM)\n - Return class + confidence\n - Modern interface', shape='box')
dot.edge('H', 'H1', style='dashed')

# Style the nodes
dot.attr('node', style='filled')
dot.node('A', fillcolor='lightblue')
dot.node('B', fillcolor='lightgreen')
dot.node('C', fillcolor='lightyellow')
dot.node('D', fillcolor='lightyellow')
dot.node('E', fillcolor='lightyellow')
dot.node('F', fillcolor='lightyellow')
dot.node('G', fillcolor='lightyellow')
dot.node('H', fillcolor='lightcoral')
dot.node('I', fillcolor='lightgray')

# Render and save in the MJ directory
output_path = os.path.join(base_dir, 'workflow_diagram')
dot.render(output_path, format='png', cleanup=True)
print(f"Diagram saved as '{output_path}.png'")
