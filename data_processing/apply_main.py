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
    yeast_num = [1,4,6]
    for i in yeast_num:
        # main(str(i))
        apply_smote.main(str(i))
        apply_smote_enn.main(str(i))
        apply_borderline_smote.main(str(i))
        apply_hybrid_sampling.main(str(i))
        apply_oversampling.main(str(i))
        apply_undersampling.main(str(i))
    print("Done")