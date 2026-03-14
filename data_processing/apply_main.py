# from apply_borderline_smote import main as apply_borderline_smote
# from apply_hybrid_sampling import main as apply_hybrid_sampling 
# from apply_oversampling  import main as apply_oversampling
# from apply_smote_enn import main as apply_smote_enn
# from apply_smote import main as apply_smote
# from apply_undersampling import main as apply_undersampling

import apply_borderline_smote
import apply_hybrid_sampling
import apply_oversampling
import apply_smote
import apply_smote_enn
import apply_undersampling


if __name__ == '__main__':

    dataset = ["yeast1-5-fold", "yeast3-5-fold", "yeast4-5-fold", "yeast6-5-fold","ecoli-0-1-3-7_vs_2-6-5-fold","glass5-5-fold","phoneme-5-fold","shuttle-c0-vs-c4-5-fold"]
    
    for dataset_name in dataset:
        print(f"Processing dataset: {dataset_name}")
        base_path = r"D:\A1-Literature-survey\dataset"

        apply_smote.main(base_path, dataset_name)
        apply_smote_enn.main(base_path, dataset_name)
        apply_borderline_smote.main(base_path, dataset_name)
        apply_hybrid_sampling.main(base_path, dataset_name)
        apply_oversampling.main(base_path, dataset_name)
        apply_undersampling.main(base_path, dataset_name)

    print("Done")