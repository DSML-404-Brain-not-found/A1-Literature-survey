import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix)

def load_dat_file(filepath):
    """Parses Kepler/ARFF style dat files."""
    data = []
    with open(filepath, 'r') as f:
        in_data_section = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith('@data'):
                in_data_section = True
                continue
            if in_data_section and not line.startswith('@'):
                parts = line.split(',')
                # Last part is the label
                features = [float(x.strip()) for x in parts[:-1]]
                label = parts[-1].strip()
                # Map positive -> 1, negative -> 0
                label_val = 1 if label.lower() == 'positive' else 0
                data.append(features + [label_val])
    
    df = pd.DataFrame(data)
    if df.empty:
        return np.array([]), np.array([])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def main():
    base_path = "/home/peng1/A1-Literature-survey/dataset/yeast1-5-fold"
    
    # Store metrics for each fold
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': [],
        'pr_auc': [],
        'g_mean': [],
        'bacc': []
    }
    
    # Random seed is typically not used in exact KNN (as there is no inherent randomness),
    # but we will define it here to match the structure of the other scripts.
    random_seed = 8
    
    for i in range(1, 6):
        train_file = os.path.join(base_path, f"yeast1-5-{i}tra.dat")
        test_file = os.path.join(base_path, f"yeast1-5-{i}tst.dat")
        
        # Load data
        X_train, y_train = load_dat_file(train_file)
        X_test, y_test = load_dat_file(test_file)
        
        # Initialize KNN
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        
        # Predict
        y_pred = knn.predict(X_test)
        y_prob = knn.predict_proba(X_test)[:, 1] # Probability for positive class
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # AUCs
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
        except ValueError: # In case only one class is present in y_test
            roc_auc = np.nan
            pr_auc = np.nan
            
        bacc = balanced_accuracy_score(y_test, y_pred)
        
        # G-mean
        # G-mean = sqrt(Sensitivity * Specificity)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        g_mean = np.sqrt(sensitivity * specificity)
        
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1_score'].append(f1)
        metrics['roc_auc'].append(roc_auc)
        metrics['pr_auc'].append(pr_auc)
        metrics['g_mean'].append(g_mean)
        metrics['bacc'].append(bacc)
        
        print(f"--- Fold {i} ---")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, G-mean: {g_mean:.4f}, BAcc: {bacc:.4f}\n")
        
    print("=== Final Average Across 5 Folds ===")
    for metric_name, values in metrics.items():
        avg_val = np.nanmean(values)
        print(f"{metric_name.capitalize():<10s}: {avg_val:.4f}")

if __name__ == "__main__":
    main()
