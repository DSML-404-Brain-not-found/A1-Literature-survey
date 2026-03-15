import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Trains the model and returns evaluation metrics for a single fold."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Probabilities are needed for AUC, fall back to decision_function or dummy if unavailable
    if hasattr(model, "predict_proba"):
        try:
            # 防呆機制：確保取得正類的預測機率
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = y_pred
    else:
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
    except ValueError:
        roc_auc = np.nan
        pr_auc = np.nan
        
    bacc = balanced_accuracy_score(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    g_mean = np.sqrt(sensitivity * specificity)

    y_true_str = json.dumps(y_test.tolist())
    y_score_str = json.dumps(y_prob.tolist())

    return acc, prec, rec, f1, roc_auc, pr_auc, g_mean, bacc, y_true_str, y_score_str

def run_rf(X_train, y_train, X_test, y_test, random_seed=8):
    model = RandomForestClassifier(random_state=random_seed)
    return evaluate_model(model, X_train, y_train, X_test, y_test)

def run_svm(X_train, y_train, X_test, y_test, random_seed=8):
    model = SVC(probability=True, random_state=random_seed)
    return evaluate_model(model, X_train, y_train, X_test, y_test)

def run_knn(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier()
    return evaluate_model(model, X_train, y_train, X_test, y_test)
