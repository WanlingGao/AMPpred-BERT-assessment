from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,matthews_corrcoef, roc_auc_score
import numpy as np
import pandas as pd
# Load real tags and test tags from the file
def load_labels(file_path):
    with open(file_path, 'r') as file:
        labels = [int(line.strip()) for line in file]
    return labels

# file path
true_labels_file = 'label.txt'
predicted_labels_file = 'pred.txt'

# Load the real label and the test label
true_labels = load_labels(true_labels_file)
predicted_labels = load_labels(predicted_labels_file)
prod=pd.read_csv('prob.txt',header=None)
print(prod)



# Computational confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
tn, fp, fn, tp = conf_matrix.ravel()

# Computational Sensitivity
sensitivity = tp / (tp + fn)
print(f'Sensitivity: {sensitivity:.4f}')

# Computational Specificity
specificity = tn / (tn + fp)
print(f'Specificity: {specificity:.4f}')

# Calculation Precision
precision = tp / (tp + fp)
print(f'Precision: {precision:.4f}')

# Calculation Accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f'Accuracy: {accuracy:.4f}')

# Calculation F1-Score
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
print(f'F1 Score: {f1:.4f}')

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(true_labels, predicted_labels)
print(f'MCC: {mcc:.4f}')

roc_auc = roc_auc_score(true_labels, prod)
print(f'auc: {roc_auc:.4f}')