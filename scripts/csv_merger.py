import pandas as pd
import re

def process_and_merge_datasets(file_paths):
    """
    Reads, cleans, and merges a list of CSV file paths.
    Extracts the source directory (e.g., '618D') from the path and adds it as a column.
    """
    merged_dataframes = []
    
    # Columns to drop based on your logic
    column_to_drop = [
        'Capture Time', 'Grade', 'NG', 'Short SN Pairing', 'Full SN', 
        '???.1', '???.2', '???_Focus', '???_Focus.1', 'AA?_??S', 'AA?_??S2', 
        'AA?_??T', 'AA?_??T2', 'AA?_??S.1', 'AA?_??T.1', 'AA?_??S.2', 'AA?_??T.2', 
        'AA?_??S.3', 'AA?_??T.3', 'AA?_??S.4', 'AA?_??T.4', 'AA?_Tilt-X', 
        'AA?_Tilt-Y', 'AA?_OC-X', 'AA?_OC-Y', 'AA?_???X', 'AA?_???Y', 
        'AA?_??0.5F-S', 'AA?_??0.5F-T', 'AA?_??0.5F-S.1', 'AA?_??0.5F-T.1', 
        'AA?_??0.5F-S.2', 'AA?_??0.5F-T.2', 'AA?_??0.5F-S.3', 'AA?_??0.5F-T.3', 
        'AA?_????', 'AA?_????.1', 'AA?_????.2', 'AA?_????.3', 'AA?_??0.5??', 
        'AA?_??0.5??.1', 'AA?_??0.5??.2', 'AA?_??0.5??.3'
    ]
    
    pos_columns = ['pos 1', 'pos 3', 'pos 5', 'pos 7', 'pos 9']

    for path in file_paths:
        # Extract the identifier (e.g., '618D') from the file path
        match = re.search(r'(\d{3}D)', path)
        source_id = match.group(1) if match else "Unknown"
        
        try:
            df = pd.read_csv(path)
            
            df = df.rename(columns={
                "????": "Capture Time", 
                "??": "Grade", 
                "NG??": "NG", 
                "???": "SN", 
                "??.1": "Short SN Pairing"
            })
            
            df = df.drop(columns=column_to_drop, errors='ignore')
            df = df.fillna("o")
            
            for col in pos_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.split('/').str[0].str.strip()
            
            if 'SN' in df.columns:
                df['SN'] = df['SN'].astype(str).str.split('/').str[0].str.strip().str.upper().str.replace(r'\.0$', '', regex=True)
            
            df['Source_ID'] = source_id
            
            merged_dataframes.append(df)
            print(f"Successfully processed: {source_id} from {path}")
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    if merged_dataframes:
        final_merged_df = pd.concat(merged_dataframes, ignore_index=True)
        return final_merged_df
    else:
        print("No dataframes to merge.")
        return pd.DataFrame()

# ==========================================
# 1. Define Paths & Merge
# ==========================================
csv_paths = [
    "/home/vilota/566-qa-2/618D/IMG/MTF数据/结果数据-总检2026-04-29-combine(结果数据 (整理结果数据）).csv",
    "/home/vilota/566-qa-2/619D/IMG/MTF数据表2026-05-07/结果数据-201个总检-2026-05-07 (1)(结果数据).csv",
    "/home/vilota/566-qa-2/620D/IMG/1号机汇总结果数据检-mingjie(结果数据).csv",
    "/home/vilota/566-qa-2/620D/IMG/2号机汇总结果数据检-mingjie(结果数据).csv",
]

final_df = process_and_merge_datasets(csv_paths)
pos_columns = ['pos 1', 'pos 3', 'pos 5', 'pos 7', 'pos 9']

if not final_df.empty:
    output_filename = "/home/vilota/566-qa-2/merge/final_merged_dataset.csv"
    final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ Success! The merged data has been saved to: {output_filename}")
    print(f"Total rows in merged dataset: {len(final_df)}")

# ==========================================
# 2. Filter out rows with unlisted letters
# ==========================================
if not final_df.empty:
    # List of valid, expected values (includes 'n' just in case)
    allowed_letters = ['o', 'xf', 'mf', 'sf', 'sn', 'mn', 'xn', 'n']
    existing_pos_cols = [col for col in pos_columns if col in final_df.columns]
    
    initial_row_count = len(final_df)
    
    # Keep only the rows where every position column has a value in 'allowed_letters'
    for col in existing_pos_cols:
        final_df = final_df[final_df[col].isin(allowed_letters)]
        
    filtered_row_count = len(final_df)
    rows_removed = initial_row_count - filtered_row_count
    
    print("\n" + "="*50)
    print(f"FILTERING: Removed {rows_removed} rows containing unlisted letters.")
    print(f"Remaining valid rows: {filtered_row_count}")
    print("="*50)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

# ==========================================
# 3. Map specific values to 'o' and 'n'
# ==========================================
if not final_df.empty:
    value_mapping = {
        'xf': 'f',
        'mf': 'f',
        'sf': 'o',
        'mn': 'n',
        'xn': 'n'
    }
    
    for col in existing_pos_cols:
        final_df[col] = final_df[col].replace(value_mapping)
            
    print("\n" + "="*50)
    print("Values remapped successfully ('xf', 'mf'-> 'f' | 'sf'->'o' | 'mn', 'xn' -> 'n')!")
    print("="*50)
    
    remapped_filename = "/home/vilota/566-qa-2/merge/remapped_merged_dataset.csv"
    final_df.to_csv(remapped_filename, index=False, encoding='utf-8-sig')
    print(f"✅ The newly remapped dataset has been saved to: {remapped_filename}")

# ==========================================
# 4. Print total counts for each letter
# ==========================================
if not final_df.empty and existing_pos_cols:
    print("\n" + "="*50)
    print("TOTAL COUNTS FOR EACH LETTER (ALL POSITIONS COMBINED)")
    print("="*50)
    
    total_counts = final_df[existing_pos_cols].stack().value_counts()
    print(total_counts.to_string())
    print("="*50 + "\n")