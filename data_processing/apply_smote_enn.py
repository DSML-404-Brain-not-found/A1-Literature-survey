import os
import shutil
import pandas as pd
from imblearn.combine import SMOTEENN
import warnings

# 忽略警告訊息
warnings.filterwarnings('ignore')

def smote_enn_keel_data(filepath, output_filepath):
    """
    讀取 KEEL 格式資料，對特徵進行 SMOTE+ENN，再輸出為相同格式
    """
    # 讀取原始檔案
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    data_start = 0
    headers = []
    
    # 找到 @data 的位置並分離 metadata（標頭資訊）
    for i, line in enumerate(lines):
        if line.strip().lower() == '@data':
            data_start = i + 1
            headers.append(line)
            break
        headers.append(line)
        
    # 解析數據行
    data_lines = lines[data_start:]
    data = []
    for line in data_lines:
        if line.strip():
            row = [x.strip() for x in line.strip().split(',')]
            data.append(row)
            
    if not data:
        print(f"  -> {filepath} 沒有資料。")
        return
        
    # 轉換成 DataFrame
    df = pd.DataFrame(data)
    
    # 分離特徵 (X) 與標籤 (y)
    # 假設最後一行為 y，前面為 X（必須是數值型）
    X = df.iloc[:, :-1].astype(float)
    y = df.iloc[:, -1]
    
    # 初始化 SMOTEENN
    smote_enn = SMOTEENN(random_state=8)
    
    # 執行混合採樣
    X_res, y_res = smote_enn.fit_resample(X, y)
    
    # 寫入目標檔案路徑
    with open(output_filepath, 'w', encoding='utf-8') as f:
        # 寫入原始標頭
        f.writelines(headers)
        
        # 寫入過採樣+降採樣後的數據
        for i in range(len(X_res)):
            # 將數值格式化至最多 4 位小數，可保留與原始格式一致性
            features = [str(round(val, 4)) for val in X_res.iloc[i]]
            label = str(y_res.iloc[i])
            row_str = ", ".join(features) + ", " + label + "\n"
            f.write(row_str)

def main(base_path=r"d:\404-Brain-not-found\dataset", dataset_name="yeast1-5-fold"):
    # 設定指定的資料夾
    # 因為題目是 yeast1_original_5_fold，這通常在 dataset 資料夾內
    input_dir = os.path.join(base_path, dataset_name)
    output_dir = os.path.join(base_path, dataset_name + "_smote_enn")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # print(f"開始針對 {input_dir} 中的訓練集執行 SMOTE+ENN 採樣...")
    
    # 遞迴處理該資料夾下的訓練集檔案
    for filename in os.listdir(input_dir):
        # 尋找所有 .dat 的訓練集 (例如 yeast1-5-1tra.dat)
        if filename.endswith('tra.dat'): 
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # print(f"正在處理檔案: {filename}")
            try:
                smote_enn_keel_data(input_path, output_path)
                # print(f"  -> 已成功產出至: {output_path}")
            except Exception as e:
                print(f"  -> 處理檔案時發生錯誤 {filename}: {str(e)}")
        elif filename.endswith('tst.dat'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            try:
                shutil.copy2(input_path, output_path)
            except Exception as e:
                print(f"  -> 複製測試檔案時發生錯誤 {filename}: {str(e)}")

if __name__ == '__main__':

    yeast_num = [1,4,6]
    for i in yeast_num:
        main(str(i))
    print("Done")
