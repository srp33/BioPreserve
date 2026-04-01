#!/usr/bin/env python3

import argparse
import pandas as pd
import re
import numpy as np

def load_data(input_csv):
    return pd.read_csv(input_csv, low_memory=False)

def normalize_colname(c):
    return c.strip().lower()

def combine_columns(df):
    """
    Combine dataset-specific columns into final standardized columns.
    Non-destructive and idempotent.
    """
    column_mapping = [
        ("meta_menopause_status_combined", "meta_menopause_status", "meta_INFERRED_MENOPAUSAL_STATE"),
        ("meta_sex_combined", "meta_gender", "meta_SEX"),
        ("meta_age_at_diagnosis_combined", "meta_age_at_diagnosis", "meta_AGE_AT_DIAGNOSIS"),
        ("meta_chemotherapy_combined", "meta_history_neoadjuvant_treatment", "meta_CHEMOTHERAPY"),
        ("meta_histological_type_combined", "meta_histological_type", "meta_HISTOLOGICAL_SUBTYPE"),
    ]

    for final_name, col1, col2 in column_mapping:
        candidates = [c for c in (col1, col2) if c in df.columns]

        if not candidates:
            continue

        combined = df[candidates[0]]
        for c in candidates[1:]:
            combined = combined.combine_first(df[c])

        df[final_name] = combined

    return df

def map_status_to_binary(df, columns):
    cols_to_convert = [col for col in columns if col in df.columns]
    if not cols_to_convert:
        return df

    df[cols_to_convert] = df[cols_to_convert].apply(lambda col: col.astype("string").str.lower())

    def status_to_binary(val):
        if pd.isnull(val):
            return np.nan
        val = str(val).strip().lower()
        try:
            num_val = float(val)
            if num_val == 0:
                return 0
            elif num_val in [1, 2, 3]:
                return 1
        except ValueError:
            pass
        if val in {'male','positive', 'yes'}:
            return 1
        if val in {'female','negative', 'no'}:
            return 0
        return np.nan

    for col in cols_to_convert:
        df[col] = df[col].map(status_to_binary)

    return df

def map_column_with_regex(df, column_name, patterns_to_values):
    if column_name not in df.columns:
        print(f"⚠️ Column {column_name} not found, skipping regex mapping")
        return df

    def map_value(val):
        if pd.isnull(val):
            return np.nan
        
        if isinstance(val, (int, float)):
            if val in [0, 1]:
                return val
            else:
                return np.nan
            
        val_str = str(val).lower().strip()
        for pattern, mapped_val in patterns_to_values:
            if re.search(pattern, val_str):
                return mapped_val
        return np.nan

    df[column_name] = df[column_name].apply(map_value)
    return df

def one_hot_encode_age(df, age_col="meta_age_at_diagnosis", bins=(0,50,70,200), labels=("lt50","50_69","ge70"), drop_original=False):
    if age_col not in df.columns:
        return df
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    age_bins = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
    age_dummies = pd.get_dummies(age_bins, prefix=age_col, dummy_na=False, dtype=int)
    df = pd.concat([df, age_dummies], axis=1)
    if drop_original:
        df = df.drop(columns=[age_col])
    return df

def threshold_encode_age(
    df,
    age_col="meta_age_at_diagnosis_combined",
    thresholds=(35, 50, 70),
    drop_original=False
):
    if age_col not in df.columns:
        return df

    # Convert to numeric, invalid parsing → NaN
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

    for t in thresholds:
        col_name = f"meta_age_lt{t}"

        # Start with NaN
        df[col_name] = np.nan

        # Only compute where age is not null
        valid_mask = df[age_col].notna()
        df.loc[valid_mask, col_name] = (df.loc[valid_mask, age_col] < t).astype(int)

    if drop_original:
        df = df.drop(columns=[age_col])

    return df

def write_output(df, output_path):
    df.to_csv(output_path, index=False)
    print(f">>> Standardized file written to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Standardize column names and map categorical metadata")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    args = parser.parse_args()

    df = load_data(args.input)
    
    # Print metadata columns after loading
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    print(f"📌 Metadata columns after loading ({len(meta_cols)} columns):")
    print(df[meta_cols].head())  # preview first 5 rows

    target_meta_cols = ['meta_menopause_status', 'meta_INFERRED_MENOPAUSAL_STATE', 
                        'meta_gender', 'meta_SEX',
                        'meta_age_at_diagnosis', 'meta_AGE_AT_DIAGNOSIS', 
                        'meta_history_neoadjuvant_treatment', 'meta_CHEMOTHERAPY',
                        'meta_histological_type', 'meta_HISTOLOGICAL_SUBTYPE',
                        'meta_source', 'meta_er_status']
    
    # Drop all nontarget metadata columns
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    keep_cols = [c for c in meta_cols if c in target_meta_cols]
    missing_cols = set(target_meta_cols) - set(keep_cols)
    if missing_cols:
        raise ValueError("Missing metadata columns: ", missing_cols)

    drop_cols = list(set(meta_cols) - set(keep_cols))
    df = df.drop(columns=drop_cols)

    # Step 1: Combine columns based on mapping
    df = combine_columns(df)

    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    print(f"📌 Metadata columns after column combination:")
    print(meta_cols)

    # Step 1.5: Drop original columns
    standardized_cols = {
        "meta_sex_combined",
        "meta_age_at_diagnosis_combined",
        "meta_menopause_status_combined",
        "meta_chemotherapy_combined",
        "meta_histological_type_combined",
        "meta_source",
        "meta_er_status"
    }

    to_drop = [
        c for c in df.columns
        if c.startswith("meta_") and c not in standardized_cols
    ]
    df = df.drop(columns=to_drop)

    # Step 2: Map tricky columns with regex
    df = map_column_with_regex(df, 'meta_menopause_status_combined', [
        (r"indeterminate", np.nan),
        (r"pre", 0),
        (r"peri", np.nan),
        (r"post", 1)
    ])
    # df = map_column_with_regex(df, 'meta_her2_status_combined', [
    #     (r"equivocal", np.nan),
    #     (r"her2-|negative", 0),
    #     (r"her2\+|positive", 1)
    # ])
    df = map_column_with_regex(df, 'meta_histological_type_combined', [
        (r"infiltrating ductal|ductal|tubular", 0),
        (r"infiltrating lobular|lobular", 1),
        (r"medullary", 2),
        (r"metaplastic", 3),
        (r"mucinous", 4),
        (r"mixed", 5),
        (r"other|nos", np.nan)
    ])

    # Step 3: One-hot encode age
    #df = one_hot_encode_age(df, age_col="meta_age_at_diagnosis_combined", bins=(0,50,70,200), labels=("lt50","50_69","ge70"))
    df = threshold_encode_age(df)

    # Step 4: Convert selected metadata to binary
    status_cols = ['meta_sex_combined', 'meta_chemotherapy_combined']
    df = map_status_to_binary(df, status_cols)

    status_cols = ['meta_sex_combined', 'meta_chemotherapy_combined']
    df = map_status_to_binary(df, status_cols)
    
    for col in standardized_cols:
        unclassified = df[col][df[col].isnull()]
        if not unclassified.empty:
            print(f"❓ Unclassified values in {col}:")
            print(unclassified.value_counts(dropna=False))
        print(f"✅ Post-conversion unique values in {col}:")
        print(df[[col, "meta_source"]].value_counts(dropna=False))

    # Step 5: Rename _combined and drop original columns
    columns_to_drop = [
        "meta_CHEMOTHERAPY", "meta_history_neoadjuvant_treatment",
        "meta_gender", "meta_SEX",
        "meta_AGE_AT_DIAGNOSIS", "meta_age_at_diagnosis",
        "meta_menopause_status", "meta_INFERRED_MENOPAUSAL_STATE",
        "meta_histological_type", "meta_HISTOLOGICAL_SUBTYPE"
    ]

    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    rename_mapping = {
        "meta_chemotherapy_combined": "meta_chemotherapy",
        "meta_sex_combined": "meta_sex",
        "meta_age_at_diagnosis_combined": "meta_age_at_diagnosis",
        "meta_menopause_status_combined": "meta_menopause_status",
        "meta_histological_type_combined": "meta_histological_type"
    }

    df = df.rename(columns=rename_mapping)

    print("Meta cols after renaming and dropping: ")
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    print(meta_cols)

    print(df[meta_cols].head(10))
    print(df[meta_cols].value_counts(dropna=False))

    # Step 5: Write output
    write_output(df, args.output)

if __name__ == "__main__":
    main()
