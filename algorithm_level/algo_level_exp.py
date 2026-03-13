import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score
)
import os
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 8

# ──────────────────────────────────────────
# 1. 資料讀取：KEEL .dat 格式
# ──────────────────────────────────────────
def load_keel_dat(filepath):
    """讀取 KEEL .dat 檔案，回傳 X (numpy array) 和 y (0/1 array)"""
    data_lines = []
    in_data = False
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.lower() == "@data":
                in_data = True
                continue
            if in_data and line and not line.startswith("@"):
                data_lines.append(line)

    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        features = list(map(float, parts[:-1]))
        label = 1 if parts[-1].lower() == "positive" else 0
        rows.append(features + [label])

    df = pd.DataFrame(rows)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


# ──────────────────────────────────────────
# 2. 指標計算：G-Mean 另外定義
# ──────────────────────────────────────────
def g_mean_score(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return np.sqrt(sensitivity * specificity)
    return 0.0

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1_Score":  f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC":   roc_auc_score(y_true, y_prob),
        "PR_AUC":    average_precision_score(y_true, y_prob),
        "G_Mean":    g_mean_score(y_true, y_pred),
        "BAcc":      balanced_accuracy_score(y_true, y_pred),
    }


# ──────────────────────────────────────────
# 3. 跑五折實驗，回傳各指標平均
# ──────────────────────────────────────────
def run_5fold(dataset_name, dataset_dir, model, model_name):
    fold_metrics = []
    for fold in range(1, 6):
        tra_path = os.path.join(dataset_dir, f"{dataset_name}-5-{fold}tra.dat")
        tst_path = os.path.join(dataset_dir, f"{dataset_name}-5-{fold}tst.dat")

        X_train, y_train = load_keel_dat(tra_path)
        X_test,  y_test  = load_keel_dat(tst_path)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 取得少數類的機率值
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        fold_metrics.append(metrics)

    # 五折平均
    avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    return avg


# ──────────────────────────────────────────
# 4. 定義六個模型
# ──────────────────────────────────────────
def get_models():
    models = {
        # ── Baseline ──
        "RF_Baseline": RandomForestClassifier(
            random_state=RANDOM_SEED
        ),
        "XGB_Baseline": XGBClassifier(
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            use_label_encoder=False
        ),
        "AdaBoost_Baseline": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            random_state=RANDOM_SEED
        ),

        # ── Cost-Sensitive ──
        "RF_CostSensitive": RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "XGB_CostSensitive": None,  # IR 需要在執行時動態計算，見下方
        "AdaBoost_CostSensitive": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced"),
            random_state=RANDOM_SEED
        ),
    }
    return models


# ──────────────────────────────────────────
# 5. 主程式
# ──────────────────────────────────────────
def main():
    # 請修改成你的資料集根目錄路徑
    base_dir = os.path.join(os.path.dirname(__file__), "dataset")

    datasets = {
        "yeast1":  os.path.join(base_dir, "yeast1-5-fold"),
        "yeast4":  os.path.join(base_dir, "yeast4-5-fold"),
        "yeast6":  os.path.join(base_dir, "yeast6-5-fold"),
    }

    all_results = []

    for ds_name, ds_dir in datasets.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*50}")

        # 計算 IR 值（用第一折 tra 作為代表）
        X_tmp, y_tmp = load_keel_dat(
            os.path.join(ds_dir, f"{ds_name.replace('yeast', 'yeast')}-5-1tra.dat")
        )
        n_neg = np.sum(y_tmp == 0)
        n_pos = np.sum(y_tmp == 1)
        ir = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"  IR = {ir:.2f}  (negative: {n_neg}, positive: {n_pos})")

        models = get_models()
        # XGBoost cost-sensitive：動態設定 scale_pos_weight = IR
        models["XGB_CostSensitive"] = XGBClassifier(
            scale_pos_weight=ir,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            use_label_encoder=False
        )

        # 資料集名稱對應到檔案前綴（yeast1 → yeast1, yeast4 → yeast4 ...）
        file_prefix = ds_name  # e.g. "yeast1"

        for model_name, model in models.items():
            avg_metrics = run_5fold(file_prefix, ds_dir, model, model_name)
            row = {"Dataset": ds_name, "Model": model_name}
            row.update(avg_metrics)
            all_results.append(row)

            print(f"\n  [{model_name}]")
            for metric, val in avg_metrics.items():
                print(f"    {metric:10s}: {val:.4f}")

    # ── 輸出成 CSV ──
    df_results = pd.DataFrame(all_results)
    output_path = os.path.join(os.path.dirname(__file__), "results.csv")
    df_results.to_csv(output_path, index=False)
    print(f"\n\n結果已儲存至：{output_path}")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()