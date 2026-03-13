import os
import shutil
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import warnings

# 忽略警告訊息
warnings.filterwarnings('ignore')

def undersampling_keel_data(filepath, output_filepath):
    """
    讀取 KEEL 格式資料，對特徵進行 Undersampling，再輸出為相同格式
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
    
    # 初始化 RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    
    # 執行降採樣
    X_res, y_res = rus.fit_resample(X, y)
    
    # 寫入目標檔案路徑
    with open(output_filepath, 'w', encoding='utf-8') as f:
        # 寫入原始標頭
        f.writelines(headers)
        
        # 寫入採樣後的數據
        for i in range(len(X_res)):
            # 將數值格式化至最多 4 位小數，可保留與原始格式一致性
            features = [str(round(val, 4)) for val in X_res.iloc[i]]
            label = str(y_res.iloc[i])
            row_str = ", ".join(features) + ", " + label + "\n"
            f.write(row_str)

def main(  yeast_num):
    # 設定指定的資料夾
    input_dir = r"d:\404-Brain-not-found\dataset\yeast"+yeast_num+"-5-fold"
    output_dir = r"d:\404-Brain-not-found\dataset\undersampling\yeast"+yeast_num
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 遞迴處理該資料夾下的訓練集檔案
    for filename in os.listdir(input_dir):
        # 尋找所有 .dat 的訓練集
        if filename.endswith('tra.dat'): 
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                undersampling_keel_data(input_path, output_path)
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
    yeast_num = "4"
    main(yeast_num)
    print("Done")
