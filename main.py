from experiment import run_experiment

if __name__ == "__main__":
    
    # method = "-"
    def yeast():
        yeast_num = ["1","4","6"]
        
        # model : "Borderline-SMOTE", "hybrid_sampling","oversampling","SMOTE","SMOTE+ENN","undersampling"
        # 使用以下變數的 True / False 來決定要執行哪些模型
        for method in ["Borderline-SMOTE", "hybrid_sampling","oversampling","SMOTE","SMOTE+ENN","undersampling"]:
            for yeast in yeast_num:
                csv_name =  method + "_result_yeast" + yeast
                file_path = r'D:\A1-Literature-survey\dataset'
                file_path = file_path + '\\' + method + r'\yeast' + yeast
                # file_path = r"D:\A1-Literature-survey\dataset\SMOTE\yeast"+yeast
                csv_path = r"D:\A1-Literature-survey\data-level-result"
                run_experiment(run_knn_model=True, run_rf_model=True, run_svm_model=True, file_path=file_path,csv_name=csv_name, csv_path=csv_path,method=method,yeast=yeast)

    def original():
            method = "original"
            csv_path = r"D:\A1-Literature-survey\data-level-result"
        
            csv_name =  method + "_result_yeast1"
            yeast="1"
            file_path = r'D:\A1-Literature-survey\dataset\yeast1-5-fold'
            run_experiment(run_knn_model=True, run_rf_model=True, run_svm_model=True, file_path=file_path,csv_name=csv_name, csv_path=csv_path,method=method,yeast=yeast)

            csv_name =  method + "_result_yeast4"
            yeast="4"
            file_path = r'D:\A1-Literature-survey\dataset\yeast4-5-fold'
            run_experiment(run_knn_model=True, run_rf_model=True, run_svm_model=True, file_path=file_path,csv_name=csv_name, csv_path=csv_path,method=method,yeast=yeast)
            
            csv_name =  method + "_result_yeast6"
            yeast="6"
            file_path = r'D:\A1-Literature-survey\dataset\yeast6-5-fold'
            run_experiment(run_knn_model=True, run_rf_model=True, run_svm_model=True, file_path=file_path,csv_name=csv_name, csv_path=csv_path,method=method,yeast=yeast)

if __name__ == "__main__":
    # original()
    yeast()