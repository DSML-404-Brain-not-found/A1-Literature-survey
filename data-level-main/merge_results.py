import pandas as pd
import os
import glob

def merge_results():
    result_dir = r"d:\A1-Literature-survey\data-level-result"
    all_files = glob.glob(os.path.join(result_dir, "*.csv"))
    
    # Exclude files that start with "merged_" to avoid re-merging previous outputs
    all_files = [f for f in all_files if not os.path.basename(f).startswith("merged_")]
    
    if not all_files:
        print("No CSV files found in", result_dir)
        return
        
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        
        # Save to single CSV
        combined_csv = os.path.join(result_dir, "merged_all_results.csv")
        combined_df.to_csv(combined_csv, index=False)
        print(f"Saved merged full CSV: {combined_csv}")
        
        # Save to Excel with sheets by Method
        output_file = os.path.join(result_dir, "merged_results.xlsx")
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for method, method_df in combined_df.groupby("Method"):
                    # Generate safe sheet name
                    sheet_name = str(method)[:31].replace('[', '').replace(']', '').replace(':', '').replace('*', '').replace('?', '').replace('/', '').replace('\\', '')
                    method_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Saved merged Excel with separate sheets by method: {output_file}")
        except Exception as e:
            print(f"Could not save Excel file (is openpyxl installed?): {e}")
            print("To install openpyxl, run: pip install openpyxl")

if __name__ == "__main__":
    merge_results()
