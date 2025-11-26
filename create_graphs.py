import json
import matplotlib.pyplot as plt
import numpy as np
import os

FILES = {
    "Single Fault": "results_single_fault.json",
    "Three Faults": "results_three_faults.json"
}

def load_data(filename):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None
    with open(filename, 'r') as f:
        return json.load(f)

# Draws the TP/FP/FN/TN boxes
def plot_confusion_matrices(data_dict):

    fig, axes = plt.subplots(1, len(data_dict), figsize=(12, 5))
    if len(data_dict) == 1: axes = [axes] # Handle single case

    for ax, (name, data) in zip(axes, data_dict.items()):
        counts = data['counts']
        # Confusion Matrix Layout
        # [ TP  FP ]
        # [ FN  TN ]
        matrix = np.array([[counts['tp'], counts['fp']], 
                           [counts['fn'], counts['tn']]])
        
        ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
        
        for i in range(2):
            for j in range(2):
                ax.text(x=j, y=i, s=matrix[i, j], va='center', ha='center', size='xx-large')
        
        ax.set_title(f"{name} Results", fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xticklabels(['', 'Fault', 'Normal'])
        ax.set_yticklabels(['', 'Fault', 'Normal'])

    plt.tight_layout()
    plt.savefig('graph_confusion_matrices.png')
    print("Saved graph_confusion_matrices.png")
    plt.show()

# Compares Precision, Recall, and F1
def plot_metrics(data_dict):
   
    labels = list(data_dict.keys())
    precision = [d['metrics']['precision'] for d in data_dict.values()]
    recall = [d['metrics']['recall'] for d in data_dict.values()]
    f1 = [d['metrics']['f1_score'] for d in data_dict.values()]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, precision, width, label='Precision')
    rects2 = ax.bar(x, recall, width, label='Recall')
    rects3 = ax.bar(x + width, f1, width, label='F1 Score')

    ax.set_ylabel('Score (0.0 - 1.0)')
    ax.set_title('Detector Performance by Fault Type')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add labels on top of bars
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    ax.bar_label(rects3, padding=3, fmt='%.2f')

    plt.tight_layout()
    plt.savefig('graph_metrics_comparison.png')
    print("Saved graph_metrics_comparison.png")
    plt.show()

loaded_data = {}
for name, fname in FILES.items():
    res = load_data(fname)
    if res:
        loaded_data[name] = res

if loaded_data:
    plot_confusion_matrices(loaded_data)
    plot_metrics(loaded_data)
else:
    print("No results files found to graph.")