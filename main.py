from experiment import run_experiment

if __name__ == "__main__":
    file_path = r"D:\404-Brain-not-found"
    csv_name = "result"
    method = "-"
    yeast_num = "1"
    # yeast_num = "4"
    # yeast_num = "6"
    
    # 使用以下變數的 True / False 來決定要執行哪些模型
    run_experiment(run_knn_model=True, run_rf_model=True, run_svm_model=True, file_path=file_path,csv_name=csv_name,method=method,yeast=yeast_num)