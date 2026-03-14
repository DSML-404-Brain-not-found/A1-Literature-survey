from experiment import run_experiment
import os

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
            file_path = r'D:\A1-Literature-survey\dataset\yeast1-5'
            run_experiment(run_knn_model=True, run_rf_model=True, run_svm_model=True, file_path=file_path,csv_name=csv_name, csv_path=csv_path,method=method,yeast=yeast)

            csv_name =  method + "_result_yeast4"
            yeast="4"
            file_path = r'D:\A1-Literature-survey\dataset\yeast4-5'
            run_experiment(run_knn_model=True, run_rf_model=True, run_svm_model=True, file_path=file_path,csv_name=csv_name, csv_path=csv_path,method=method,yeast=yeast)
            
            csv_name =  method + "_result_yeast6"
            yeast="6"
            file_path = r'D:\A1-Literature-survey\dataset\yeast6-5'
            run_experiment(run_knn_model=True, run_rf_model=True, run_svm_model=True, file_path=file_path,csv_name=csv_name, csv_path=csv_path,method=method,yeast=yeast)

    def new_main():
        # dataset = ["yeast1-5", "yeast3-5", "yeast4-5", "yeast6-5","ecoli-0-1-3-7_vs_2-6-5","glass5-5","phoneme-5","shuttle-c0-vs-c4-5"]
        dataset = ["yeast1-5", "yeast3-5", "yeast4-5", "yeast6-5","glass5-5","phoneme-5","shuttle-c0-vs-c4-5","poker-8-9_vs_6-5"]
        dataset_folder = ["yeast1-5-fold_", "yeast3-5-fold_", "yeast4-5-fold_", "yeast6-5-fold_","glass5-5-fold_","phoneme-5-fold_","shuttle-c0-vs-c4-5-fold_","poker-8-9_vs_6-5-fold_"]
        method_list = ["original", "Borderline-SMOTE", "hybrid_sampling", "oversampling", "SMOTE", "SMOTE+ENN", "undersampling"]
        
        # Mapping from method name to folder suffix
        method_to_suffix = {
            "original": "",
            "Borderline-SMOTE": "borderline_smote",
            "hybrid_sampling": "hybrid_sampling",
            "oversampling": "oversampling",
            "SMOTE": "smote",
            "SMOTE+ENN": "smote_enn",
            "undersampling": "undersampling",
        }

        base_path = r'D:\A1-Literature-survey\dataset'
        csv_path = r"D:\A1-Literature-survey\data-level-result"

        for current_method in method_list:
            for i in range(len(dataset)):
                # Construct the correct directory name from the base folder and the method's specific suffix
                folder_name = dataset_folder[i] + method_to_suffix[current_method]
                file_path = os.path.join(base_path, folder_name)
                
                csv_name = f"{current_method}_result_{dataset[i]}"
                run_experiment(run_knn_model=True, run_rf_model=True, run_svm_model=True, file_path=file_path, csv_name=csv_name, csv_path=csv_path, method=current_method, dataset=dataset[i])

if __name__ == "__main__":
    # original()
    # yeast()
    new_main()