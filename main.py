import os
import sys
import numpy as np
import pandas as pd
from models import evaluate_model
from experiment import run_experiment
from evaluator import load_dat_file
# ── 將 hybrid_methods 資料夾加入 import 路徑 ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hybrid_methods"))
from hybrid_methods import StandardAdaBoost, SMOTEBoost, RUSBoost, RHSBoost, SUBoost, SMOTECSL

import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 8

# ----------------------------------------------------------
# Data-level Methods
# ----------------------------------------------------------
if __name__ == "__main__":
    file_path = r"D:\404-Brain-not-found"
    csv_name = "result"
    method = "-"
    yeast_num = "1"
    # yeast_num = "4"
    # yeast_num = "6"
    
    # 使用以下變數的 True / False 來決定要執行哪些模型
    run_experiment(run_knn_model=True, run_rf_model=True, run_svm_model=True, file_path=file_path,csv_name=csv_name,method=method,yeast=yeast_num)


# ----------------------------------------------------------
# Hybrid Methods
# ----------------------------------------------------------
def run_hybrid_methods_pipeline(
    base_names,
    path,
    model_name,
    model_class,
    random_state=RANDOM_SEED,
    **kwargs,
):
    """
    Run evaluation for a given hybrid method on multiple datasets.

    Parameters
    ----------
    base_names : list[str]
        資料集基礎名稱（e.g. ["yeast1", "yeast4", "yeast6"]）。
        對應 <path>/<base_name>-5-fold/ 子資料夾。
    path : str
        包含各資料集子資料夾的根目錄。
    model_name : str
        混合方法名稱，用於日誌。
    model_class : type
        混合方法類別（需符合 sklearn API：fit / predict / predict_proba）。
    random_state : int
        固定隨機種子，預設為 8（與 experiment.py 一致）。
    **kwargs
        傳遞給 model_class(...) 的額外參數。

    Returns
    -------
    list[dict]
        每個 dataset 的平均指標列（含 Method、Model、Dataset 欄位）。
    """
    print("\n" + "# " + 30 * "-" + f" {model_name} " + "-" * 30 + " #")

    metric_keys = [
        "Accuracy", "Precision", "Recall", "F1_Score",
        "ROC_AUC", "PR_AUC", "G_Mean", "BAcc",
    ]

    collected_rows = []  # 回傳值：各 dataset 的平均指標列

    # ── 逐資料集處理（對應 experiment.py: evaluate_models 迴圈）─────────────
    for base_name in base_names:
        print(f"  [{model_name}] Dataset: {base_name}")

        # 5-fold 交叉驗證；資料放在 <path>/<base_name>-5-fold/
        fold_dir = os.path.join(path, f"{base_name}-5-fold")

        all_results = {k: [] for k in metric_keys}
        fold_rows = []  # 記錄每個 fold 的原始指標列

        for fold in range(1, 6):
            tra_file = os.path.join(fold_dir, f"{base_name}-5-{fold}tra.dat")
            tst_file = os.path.join(fold_dir, f"{base_name}-5-{fold}tst.dat")

            X_train, y_train = load_dat_file(tra_file)
            X_test,  y_test  = load_dat_file(tst_file)

            if len(X_train) == 0 or len(X_test) == 0:
                print(f"    警告：Fold {fold} 資料為空，跳過。")
                continue

            model = model_class(random_state=random_state, **kwargs)
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
            # metrics = (acc, prec, rec, f1, roc_auc, pr_auc, g_mean, bacc)

            row = {"Fold": fold}
            for k, v in zip(metric_keys, metrics):
                row[k] = v
                all_results[k].append(v)
            fold_rows.append(row)
            print(f"    Fold {fold} 完成")

        if not fold_rows:
            print(f"    警告：{base_name} 無有效結果，跳過。")
            continue

        # ── 計算 5-fold 平均（對應 experiment.py: final_averages）────────────
        avg_metrics = {}
        for k in metric_keys:
            avg_metrics[k] = np.nanmean(all_results[k]) if all_results[k] else np.nan

        # ── 印出平均摘要 ──────────────────────────────────────────────────
        print(f"    5-Fold Average ({base_name}):")
        for k in metric_keys:
            print(f"      {k:12s}: {avg_metrics[k]:.4f}")

        # ── 記錄此 dataset 的平均結果（含 Method、Model、Dataset 欄位）────
        row_entry = {
            "Method":  model_name,
            "Model":   "-",
            "Dataset": base_name,
        }
        row_entry.update(avg_metrics)
        collected_rows.append(row_entry)

    return collected_rows


if __name__ == "__main__":
    # ── 資料集根目錄 ────
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

    # ── 要評估的資料集清單 ─────────────────────────────────────────────────
    base_names = [
        "yeast1",
        "yeast3",
        "yeast4",
        "yeast6",
    ]

    # ── 混合方法字典：名稱 → (類別, 參數) ────────────────────────────────
    hybrid_methods = {
        "StandardAdaBoost": (StandardAdaBoost, {"n_estimators": 100}),
        "SMOTEBoost": (SMOTEBoost, {"n_estimators": 100, "k_neighbors": 3}),
        "RUSBoost":   (RUSBoost,   {"n_estimators": 100}),
        "RHSBoost":   (RHSBoost,   {"n_estimators": 100, "k_neighbors": 5}),
        "SUBoost":    (SUBoost,    {"n_estimators": 100}),
        "SMOTECSL":   (SMOTECSL,   {"k_neighbors": 5}),
    }

    # ── 逐方法執行 Pipeline，收集所有結果 ────────────────────────────────
    all_rows = []
    for method_name, (model_class, params) in hybrid_methods.items():
        rows = run_hybrid_methods_pipeline(
            base_names,
            path,
            method_name,
            model_class,
            random_state=RANDOM_SEED,
            **params,
        )
        all_rows.extend(rows)

    # ── 組合最終 DataFrame，欄位順序依需求 ───────────────────────────────
    metric_col_map = {
        "Accuracy":  "Accuracy",
        "Precision": "Precision",
        "Recall":    "Recall",
        "F1_Score":  "F1-score",
        "ROC_AUC":   "ROC-AUC",
        "PR_AUC":    "PR-AUC",
        "G_Mean":    "G-Mean",
        "BAcc":      "BAcc",
    }

    output_rows = []
    for row in all_rows:
        method = row.get("Method", "-")
        model  = row.get("Model",  "-")
        # Methods+Model：若其中一個為 "-"，取另一個有名稱的欄位
        if method == "-" and model == "-":
            methods_model = "-"
        elif method == "-":
            methods_model = model
        elif model == "-":
            methods_model = method
        else:
            methods_model = f"{method}+{model}"

        dataset = row.get("Dataset", "-")
        out = {
            "Method":        method,
            "Model":         model,
            "Methods+Model": methods_model,
            "Dataset":       dataset,
        }
        for src_key, dst_key in metric_col_map.items():
            out[dst_key] = row.get(src_key, np.nan)
        output_rows.append(out)

    final_df = pd.DataFrame(output_rows, columns=[
        "Method", "Model", "Methods+Model", "Dataset",
        "Accuracy", "Precision", "Recall", "F1-score",
        "ROC-AUC", "PR-AUC", "G-Mean", "BAcc",
    ])

    # ── 輸出統一結果 CSV ──────────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid_method_result")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "hybrid_results.csv")
    final_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n所有方法的結果已統一儲存至：{csv_path}")