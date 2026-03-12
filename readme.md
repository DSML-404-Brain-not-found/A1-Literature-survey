# 模型實驗整合架構 (Model Experiment Framework)

這個專案是為了解析 `yeast` 資料集的 `.dat` 格式內容，並使用五折交叉驗證（5-fold cross-validation）來評估多種分類機器學習模型（包含 KNN, Random Forest, 與 SVM）效能的小型整合框架。

## 檔案與模組說明

### 1. `main.py`
這是使用者主要的執行入口腳本。
- **作用：** 提供最乾淨與簡短的介面。使用者只需要在這個檔案中修改對應的參數（布林值 `True` / `False`），即可決定這次實驗要跑哪些模型，並設定資料集的路徑與輸出 CSV 檔名。
- **使用方法：** 直接執行 `python main.py`

### 2. `experiment.py`
這是連接 `main.py` 介面與實際模型評估邏輯的中間層腳本。
- **函式 `run_experiment()`：**
  讀取從 `main.py` 傳入的模型觸發開關（`run_knn_model`, `run_rf_model`, `run_svm_model`）與路徑設定。內部會組合一個要執行的模型庫清單並呼叫 `evaluate_models`，最後負責將 Pandas DataFrame 整理並加上 `Method` 欄位後匯出成 CSV 檔案。

### 3. `evaluator.py`
資料處理與驗證的核心迴圈（Loop）。
- **函式 `load_dat_file(filepath)`：**
  負責讀取特定格式（`@data` 段落）的 `.dat` 檔案，轉換為特徵陣列 `X` (特徵包含 Mcg, Gvh, Alm, Mit, Erl, Pox, Vac, Nuc 等) 與目標陣列 `y` (分類映射 positive->1, negative->0)。
- **函式 `evaluate_models(base_path, models_to_run, random_seed)`：** 
  跑過五折（Fold 1 到 Fold 5）訓練與測試集。依次對指定的每個模型取得該折的 Accuracy, Precision, Recall, F1 score, ROC-AUC, PR-AUC, G-mean, BAcc，最終計算這五折的平均值並匯總成 DataFrame 表格。

### 4. `models.py`
統整並建立各別 sklearn 機器學習模型。
- **函式 `evaluate_model(...)`：** 傳入模型物件與切分好的資料，內部負責宣告 `.fit()` 與 `.predict()`，最後計算並回傳八個目標成效指標。
- **函式 `run_rf(...)`：** 呼叫 `evaluate_model` 執行 Random Forest Classifier。
- **函式 `run_svm(...)`：** 呼叫 `evaluate_model` 執行 Support Vector Machine。
- **函式 `run_knn(...)`：** 呼叫 `evaluate_model` 執行 K-Neighbors Classifier。

---

## 輸出結果範例
執行完 `main.py` 之後，框架會在您指定的目錄下產出一份 CSV (例如 `result.csv`)：
```csv
Method,Model,Accuracy,Precision,Recall,F1_Score,ROC_AUC,PR_AUC,G_Mean,BAcc
-,KNN,0.739219...,0.627768...,...
-,RandomForest,0.778317...,0.665044...,...
-,SVM,0.756757...,0.695627...,...
```
