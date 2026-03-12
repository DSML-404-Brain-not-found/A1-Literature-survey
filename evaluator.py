import os
import pandas as pd
import numpy as np

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

def evaluate_models(base_path, models_to_run, random_seed=8):
    """
    Evaluates specific models on 5-fold Yeast cross-validation.
    Returns a dataframe of the average metrics.
    """
    # Dictionary to hold the combined metric lists for all selected models
    all_results = {}
    
    for model_name, model_func in models_to_run.items():
        all_results[model_name] = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
            'roc_auc': [], 'pr_auc': [], 'g_mean': [], 'bacc': []
        }
    
    # Run through the 5 folds
    for i in range(1, 6):
        train_file = os.path.join(base_path, f"yeast1-5-{i}tra.dat")
        test_file = os.path.join(base_path, f"yeast1-5-{i}tst.dat")
        
        X_train, y_train = load_dat_file(train_file)
        X_test, y_test = load_dat_file(test_file)
        
        for model_name, model_func in models_to_run.items():
            # Most model funcs accept random_seed except KNN which may ignore it, 
            # We wrapped them to either use or ignore appropriately
            if model_name == 'KNN':
                metrics = model_func(X_train, y_train, X_test, y_test)
            else:
                metrics = model_func(X_train, y_train, X_test, y_test, random_seed=random_seed)
            
            all_results[model_name]['accuracy'].append(metrics[0])
            all_results[model_name]['precision'].append(metrics[1])
            all_results[model_name]['recall'].append(metrics[2])
            all_results[model_name]['f1_score'].append(metrics[3])
            all_results[model_name]['roc_auc'].append(metrics[4])
            all_results[model_name]['pr_auc'].append(metrics[5])
            all_results[model_name]['g_mean'].append(metrics[6])
            all_results[model_name]['bacc'].append(metrics[7])

    # Compute averages
    final_averages = []
    
    for model_name, metrics in all_results.items():
        avg_metrics = {
            'Model': model_name,
            'Accuracy': np.nanmean(metrics['accuracy']),
            'Precision': np.nanmean(metrics['precision']),
            'Recall': np.nanmean(metrics['recall']),
            'F1_Score': np.nanmean(metrics['f1_score']),
            'ROC_AUC': np.nanmean(metrics['roc_auc']),
            'PR_AUC': np.nanmean(metrics['pr_auc']),
            'G_Mean': np.nanmean(metrics['g_mean']),
            'BAcc': np.nanmean(metrics['bacc'])
        }
        final_averages.append(avg_metrics)
        
    return pd.DataFrame(final_averages)
