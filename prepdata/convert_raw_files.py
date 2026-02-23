# /scripts/prepdata/convert_raw_files.py

import os
import pandas as pd
import numpy as np
import gzip
import warnings
import sys
from pathlib import Path
import re
import argparse
import time
import shutil
import csv
import zipfile

# Import gene ID utilities
from gene_id_utils import detect_gene_id_type, convert_gene_ids_to_symbols

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

def print_now(*args, **kwargs):
    print(*args, flush=True, **kwargs)


def smart_read_dataframe(file_path: Path, debug: bool = False, **kwargs) -> tuple:
    """
    Attempt to read a dataframe using pandas' inference. If it fails,
    run diagnostics to suggest the cause of the error.
    
    Returns a (dataframe, status_string) tuple. On failure, the dataframe
    is None and the string contains diagnostic information.
    """
    # First, try common delimiters explicitly for TSV/CSV files
    for delimiter in ['\t', ',', ';']:
        try:
            df = pd.read_csv(
                file_path,
                delimiter=delimiter,
                compression='infer',
                **kwargs
            )
            if df is not None and not df.empty and len(df.columns) > 1:
                return df, f"Success (delimiter: '{delimiter}')"
        except Exception as e:
            if debug:
                print_now(f"DEBUG: Failed with delimiter '{delimiter}'. Error: {e}")
            continue
    
    # Fallback: try letting pandas infer everything.
    try:
        df = pd.read_csv(
            file_path,
            delimiter=None,
            engine='python',
            compression='infer',
            **kwargs
        )
        if df is not None and not df.empty:
            return df, "Success (Pandas auto-inference)"
    except Exception as e:
        if debug:
            print_now(f"DEBUG: Initial pandas read failed. Error: {e}")
        pass

    # If the first attempt fails, run diagnostics.
    diagnostics = []
    
    # Check 1: Validate file existence and size.
    if not file_path.exists():
        return None, "Failed: File does not exist."
    if file_path.stat().st_size == 0:
        return None, "Failed: File is empty."

    # Check 2: Check for compression mismatch (magic bytes vs. extension).
    is_gzip, is_zip = False, False
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            is_gzip = header.startswith(b'\x1f\x8b')
            is_zip = header.startswith(b'\x50\x4b\x03\x04')
            
        ext = file_path.suffix.lower()
        if ext == '.gz' and not is_gzip:
            diagnostics.append("File has '.gz' extension but is not a valid gzip file.")
        elif ext != '.gz' and is_gzip:
            diagnostics.append("File appears to be gzipped but lacks '.gz' extension.")
        
        if ext == '.zip' and not is_zip:
            diagnostics.append("File has '.zip' extension but is not a valid zip archive.")
        elif ext != '.zip' and is_zip:
            diagnostics.append("File appears to be a zip archive but lacks '.zip' extension.")

    except IOError as e:
        return None, f"Failed: Could not read file for diagnostics. Error: {e}"

    # Check 3: Sniff the first line for a potential delimiter.
    try:
        open_func = gzip.open if is_gzip else open
        
        if is_zip:
            diagnostics.append("Delimiter check skipped for zip archives.")
        else:
            with open_func(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(first_line, delimiters=',\t;| ')
                diagnostics.append(f"Detected potential delimiter: '{dialect.delimiter}'.")
    except (csv.Error, StopIteration):
        diagnostics.append("Could not detect a common delimiter in the first line.")
    except Exception as e:
        if debug:
            print_now(f"DEBUG: Delimiter check failed. Error: {e}")
        diagnostics.append("Could not read first line (possible encoding or corruption issue).")

    # Format the final diagnostic report.
    if not diagnostics:
        diagnostics.append("Pandas parser failed for an unknown reason. The file may be malformed.")

    return None, f"Failed. Diagnostics: {'; '.join(diagnostics)}"
    
# === DATA PROCESSING LOGIC ===

def _find_dataset_file(folder_path: Path, dataset_id: str, prefix: str) -> Path | None:
    """Find a dataset file with a given prefix, trying common extensions and patterns."""
    for ext in ['.csv', '.tsv']:
        # Try exact match first: e.g., expression_GSE12345.csv
        exact_path = folder_path / f"{prefix}{dataset_id.upper()}{ext}"
        if exact_path.exists():
            return exact_path
        # Fallback to pattern matching: e.g., expression_data_GSE12345_raw.tsv
        for candidate in folder_path.glob(f"{prefix}*{ext}"):
            if dataset_id.upper().replace('_', '') in candidate.stem.upper().replace('_', ''):
                return candidate
    return None

def find_dataset_files(raw_folder_path: Path, dataset_id: str) -> tuple[Path | None, Path | None]:
    """Find expression and metadata files for a given dataset ID."""
    expression_file = _find_dataset_file(raw_folder_path, dataset_id, "expression_")
    meta_file = _find_dataset_file(raw_folder_path, dataset_id, "meta_")
    return expression_file, meta_file

def _should_transpose(expr_df: pd.DataFrame, meta_df: pd.DataFrame | None) -> tuple[bool, str]:
    """Apply heuristics to determine if the expression dataframe needs transposing."""
    rows, cols = expr_df.shape
    
    # Heuristic 1: Compare dimensions to sample count from metadata
    if meta_df is not None:
        expected_samples = len(meta_df)
        # If columns are a much closer match to sample count than rows, transpose.
        if abs(cols - expected_samples) < abs(rows - expected_samples) * 0.5:
            return True, f"columns ({cols}) closer to expected samples ({expected_samples}) than rows ({rows})"
    
    # Heuristic 2: Genomics convention (genes >> samples)
    if rows > cols * 10:
        return True, f"many more rows ({rows}) than columns ({cols}), likely genes in rows"
    if cols > rows * 10:
        return False, f"many more columns ({cols}) than rows ({rows}), likely genes in columns"

    # Heuristic 3: Detect sample-like IDs in column names
    sample_like_cols = [c for c in expr_df.columns if isinstance(c, str) and c.startswith(('GSM', 'TCGA', 'Sample'))]
    if len(sample_like_cols) / cols > 0.5: # If >50% of columns look like samples
        return True, f"detected {len(sample_like_cols)} sample-like column names"
        
    # Heuristic 4: Detect gene-like IDs in the first column's content
    first_col_content = expr_df.iloc[:10, 0].astype(str).str.upper()
    if first_col_content.str.contains('ENSG|GENE|PROBE').sum() > 3:
        return True, f"first column '{expr_df.columns[0]}' contains gene-like identifiers"
        
    # Default Fallback: Assume genes are in rows if there are more rows than columns
    if rows > cols:
        return True, f"fallback: more rows ({rows}) than columns ({cols})"
        
    return False, "fallback: not transposing"

# Gene ID functions imported from gene_id_utils module

def process_dataset(raw_folder_path: Path, dataset_id: str, output_base_dir: Path, debug: bool):
    """Process a single dataset: combine expression and metadata, save to the output directory."""
    print_now(f"\n🔄 Processing dataset: {dataset_id}")
    
    try:
        # 1. Find and read files (this is the first validation step)
        expression_file, meta_file = find_dataset_files(raw_folder_path, dataset_id)
        if not expression_file or not meta_file:
            print_now("   ❌ Validation failed: Expression or metadata file not found. Skipping.")
            return False
            
        expr_df, _ = smart_read_dataframe(expression_file)
        meta_df, _ = smart_read_dataframe(meta_file)
        if expr_df is None or meta_df is None:
            print_now("   ❌ Validation failed: Could not read expression or metadata file. Skipping.")
            return False

        # 2. Transpose expression data if necessary
        needs_transpose, reason = _should_transpose(expr_df, meta_df)
        print_now(f"   🔍 Transpose decision: {'YES' if needs_transpose else 'NO'} - {reason}")
        if needs_transpose:
            # Identify gene info columns vs sample data columns
            gene_info_cols = []
            for col in expr_df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['dataset', 'entrez', 'gene', 'symbol', 'ensembl', 'chromosome', 'biotype']):
                    gene_info_cols.append(col)
                else:
                    break  # Stop at first non-gene-info column
            
            if gene_info_cols:
                # Find the best gene identifier column to use as column names after transpose
                gene_id_col = None
                for preferred in ['HGNC_Symbol', 'Entrez_Gene_ID', 'Ensembl_Gene_ID']:
                    if preferred in gene_info_cols:
                        gene_id_col = preferred
                        break
                if gene_id_col is None:
                    gene_id_col = gene_info_cols[1] if len(gene_info_cols) > 1 else gene_info_cols[0]
                
                # Create gene identifiers for column names
                gene_ids = expr_df[gene_id_col].astype(str)
                
                # Separate expression data and transpose
                expr_data = expr_df.drop(columns=gene_info_cols)
                expr_df = expr_data.T
                expr_df.columns = gene_ids
                expr_df.index.name = 'Sample_ID'
            else:
                # Fallback: use first column as gene index
                expr_df = expr_df.set_index(expr_df.columns[0]).T
        else:
            # Assume first column is sample ID
            expr_df = expr_df.set_index(expr_df.columns[0])
        
        # 3. Prepare metadata
        # Find and set sample ID column as index - prioritize 'Sample_ID' specifically
        sample_id_col = None
        for col in meta_df.columns:
            col_lower = str(col).lower()
            if col_lower == 'sample_id':
                sample_id_col = col
                break
            elif 'sample' in col_lower and 'id' in col_lower:
                sample_id_col = col
                break
        
        # If no sample_id column found, look for columns with GSM-like values
        if sample_id_col is None:
            for col in meta_df.columns:
                if meta_df[col].astype(str).str.contains('GSM|TCGA|Sample').any():
                    sample_id_col = col
                    break
        
        if sample_id_col is None:
            # Last resort: use second column if first is dataset_id
            if len(meta_df.columns) > 1 and 'dataset' in str(meta_df.columns[0]).lower():
                sample_id_col = meta_df.columns[1]
            else:
                sample_id_col = meta_df.columns[0]
            
        if debug:
            print_now(f"DEBUG: Using metadata column '{sample_id_col}' as sample ID")
            
        meta_df = meta_df.set_index(sample_id_col)
        # Prefix columns with 'meta_'
        meta_df = meta_df.add_prefix('meta_')
        lower_to_full_case_columns = {col.lower():col for col in meta_df.columns}

        # Standardize er status column name
        other_er_columns = ['meta_er_status', 'meta_er_ihc', 'meta_er', 'meta_er_status_diagnosis', 'meta_estrogen_receptor_status', 'meta_er_status_by_ihc', 'meta_er_status_ihc', 'meta_er_consensus', 'meta_esr1_status']
        for er_column in other_er_columns:
            if er_column in lower_to_full_case_columns:
                er_column = lower_to_full_case_columns[er_column]
                print_now(f"Using {er_column} as ER status column")
                meta_df = meta_df.rename(columns={er_column: 'meta_er_status'})
                break

        if "meta_er_status" not in meta_df.columns.tolist():
            print_now(f"   ❌ ER status column not found. ")
            print_now(f"DEBUG: Available columns: {meta_df.columns.tolist()}")
        else:
            print_now(f"Unique er values: ")
            print_now(meta_df['meta_er_status'].value_counts())

        # Standardize pr status column name
        other_pr_columns = ['meta_pr', 'meta_pr_status', 'meta_pgr_status', 'meta_pr_status_diagnosis', 'meta_progesterone_receptor_status', 'meta_pr_status_ihc', 'meta_pr_ihc', 'meta_pgr_consensus', 'meta_prihc', 'meta_pr_status_by_ihc']
        for pr_column in other_pr_columns:
            if pr_column in lower_to_full_case_columns:
                pr_column = lower_to_full_case_columns[pr_column]
                print_now(f"Using {pr_column} as PR status column")
                meta_df = meta_df.rename(columns={pr_column: 'meta_pr_status'})
                break

        if "meta_pr_status" not in meta_df.columns.tolist():
            print_now(f"   ❌ PR status column not found. ")
            print_now(f"DEBUG: Available columns: {meta_df.columns.tolist()}")
        else:
            print_now(f"Unique pr values: ")
            print_now(meta_df['meta_pr_status'].value_counts())

        # Standardize her2 status column name
        other_her2_columns = ['meta_her2', 'meta_her_2', 'meta_her2_status', 'meta_her_2_status', 'meta_her2_status_diagnosis', 'meta_her2_receptor_status', 'meta_her2_ihc', 'meta_her2_status_by_ihc', 'meta_her2_consensus', 'meta_her2_snp6']
        for her2_column in other_her2_columns:
            if her2_column in lower_to_full_case_columns:
                her2_column = lower_to_full_case_columns[her2_column]
                print_now(f"Using {her2_column} as HER2 status column")
                meta_df = meta_df.rename(columns={her2_column: 'meta_her2_status'})
                break

        if "meta_her2_status" not in meta_df.columns.tolist():
            print_now(f"   ❌ HER2 status column not found. ")
            print_now(f"DEBUG: Available columns: {meta_df.columns.tolist()}")
        else:
            print_now(f"Unique her2 values: ")
            print_now(meta_df['meta_her2_status'].value_counts())

        # Map column values to 0 for negative and 1 for positive.
        expected_cols = ['meta_er_status', 'meta_pr_status', 'meta_her2_status']
        cols_to_convert = [col for col in expected_cols if col in meta_df.columns]

        # Lowercase all values in relevant columns
        meta_df[cols_to_convert] = meta_df[cols_to_convert].apply(lambda col: col.astype(str).str.lower())

        def status_to_binary(val):
            if pd.isnull(val):
                return np.nan 

            val = str(val).strip().lower()

            try: 
                # Case 1: Already numeric
                num_val = float(val)
                if num_val == 0:
                    return 0
                elif num_val in [1, 2, 3]:
                    return 1
            except ValueError: 
                pass

            positive_vals = {'positive', 'p', 'pos', 'pos-low', '1', '2', '3', 'er+', 'he+', 'pr+', 'pgr+'}
            negative_vals = {'negative', 'n', 'neg', '0', 'er-', 'he-', 'pr-', 'pgr-'}

            for pos in positive_vals:
                if pos in val:
                    return 1
            for neg in negative_vals:
                if neg in val:
                    return 0

            return np.nan

        meta_df[cols_to_convert] = meta_df[cols_to_convert].map(status_to_binary)

        # print_now("Number of values that couldn't be classified: ", meta_df[cols_to_convert].isnull().sum())
        for col in cols_to_convert:
            unclassified = meta_df[col][meta_df[col].isnull()]
            if not unclassified.empty:
                print_now(f"❓ Unclassified values in {col}:")
                print_now(unclassified.value_counts(dropna=False))

        
        # Print new unique values in the target columns:
        for col in cols_to_convert:
            print_now(f"Post-conversion unique values in {col}:")
            print_now(meta_df[col].value_counts(dropna=False))


        # 4. Align and combine (the final and most critical validation)
        common_samples = expr_df.index.intersection(meta_df.index)
        if len(common_samples) == 0:
            print_now(f"   ❌ Validation failed: No common samples found between files. Skipping.")
            if debug:
                # DEBUG: Show sample ID examples to help diagnose mismatches.
                expr_samples = sorted(list(expr_df.index))[:5]
                meta_samples = sorted(list(meta_df.index))[:5]
                print_now(f"DEBUG: Expression sample IDs: {expr_samples}")
                print_now(f"DEBUG: Metadata sample IDs: {meta_samples}")
                print_now(f"DEBUG: Expression index type: {type(expr_df.index[0]) if len(expr_df.index) > 0 else 'empty'}")
                print_now(f"DEBUG: Metadata index type: {type(meta_df.index[0]) if len(meta_df.index) > 0 else 'empty'}")
            return False
            
        print_now(f"   🔗 Found {len(common_samples)} common samples.")
        combined_df = pd.concat([expr_df.loc[common_samples], meta_df.loc[common_samples]], axis=1)

        # Name the index "meta_Sample_ID"
        combined_df.index.name = 'meta_Sample_ID'

        # 4.5. Gene ID detection and conversion
        gene_cols = [col for col in combined_df.columns if not col.startswith('meta_')]
        if gene_cols:
            print_now(f"   🔍 Detecting gene ID type from {len(gene_cols)} gene columns...")
            detection = detect_gene_id_type(gene_cols, debug=debug)
            
            if debug:
                print_now(f"   📝 Examples: {', '.join(detection['examples'])}")
            
            # Convert gene IDs to symbols if not already symbols
            if detection['type'] != 'gene_symbol' and detection['confidence'] > 0.5:
                combined_df = convert_gene_ids_to_symbols(combined_df, detection['type'], 
                                                        annotation_dir="grp_batch_effects/data/annotations", 
                                                        debug=debug)

        # 5. Save result
        output_dir = output_base_dir / dataset_id.lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "unadjusted.csv"
        combined_df.to_csv(output_file)
        
        print_now(f"   ✅ Successfully saved: {output_file} ({output_file.stat().st_size:,} bytes)")
        return True

    except Exception as e:
        print_now(f"   ❌ Error processing {dataset_id}: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False

def verify_datasets(datasets_info, target_dir):
    """Verify processed datasets by reporting null value statistics."""
    print_now("\n" + "="*80)
    print_now("DATASET VERIFICATION - NULL VALUE ANALYSIS")
    print_now("="*80)
    
    for info in datasets_info:
        dataset_id = info['id']
        output_file = Path(target_dir) / dataset_id.lower() / "unadjusted.csv"
        
        if not output_file.exists():
            print_now(f"❌ {dataset_id}: Output file not found - {output_file}")
            continue
            
        try:
            # Read the processed dataset
            df = pd.read_csv(output_file, index_col=0)
            
            # Calculate null statistics
            total_rows, total_cols = df.shape
            rows_with_nulls = df.isnull().any(axis=1).sum()
            cols_with_nulls = df.isnull().any(axis=0).sum()
            total_nulls = df.isnull().sum().sum()
            
            # Calculate percentages
            null_row_pct = (rows_with_nulls / total_rows) * 100 if total_rows > 0 else 0
            null_col_pct = (cols_with_nulls / total_cols) * 100 if total_cols > 0 else 0
            total_null_pct = (total_nulls / (total_rows * total_cols)) * 100 if (total_rows * total_cols) > 0 else 0
            
            # Separate gene and metadata columns for detailed analysis
            meta_cols = [col for col in df.columns if col.startswith('meta_')]
            gene_cols = [col for col in df.columns if not col.startswith('meta_')]
            
            print_now(f"✅ {dataset_id}:")
            print_now(f"   📊 Shape: {total_rows} samples × {total_cols} features ({len(gene_cols)} genes + {len(meta_cols)} metadata)")
            print_now(f"   🔍 Null values: {total_nulls:,} ({total_null_pct:.1f}% of all values)")
            print_now(f"   📋 Rows with nulls: {rows_with_nulls}/{total_rows} ({null_row_pct:.1f}%)")
            print_now(f"   📋 Columns with nulls: {cols_with_nulls}/{total_cols} ({null_col_pct:.1f}%)")
            
            # Check for critical metadata columns
            required_meta = ['meta_er_status']
            missing_critical = [col for col in required_meta if col not in df.columns]
            if missing_critical:
                print_now(f"   ⚠️  Missing critical columns: {missing_critical}")
            
        except Exception as e:
            print_now(f"❌ {dataset_id}: Error reading file - {e}")

def scan_for_datasets(raw_data_dir: Path) -> list[dict]:
    """Scan the raw data directory for valid dataset folders."""
    datasets = []
    
    if not raw_data_dir.exists():
        print_now(f"❌ Raw data directory {raw_data_dir} does not exist")
        return datasets
    
    for item in raw_data_dir.iterdir():
        if item.is_dir():
            dataset_id = item.name
            expression_file, meta_file = find_dataset_files(item, dataset_id)
            
            if expression_file and meta_file:
                datasets.append({
                    'id': dataset_id,
                    'path': item,
                    'expression_file': expression_file,
                    'meta_file': meta_file
                })
            else:
                print_now(f"⚠️  Skipping {dataset_id}: missing expression or metadata file")
    
    return datasets

def main():
    """Main function to process all datasets in the raw data directory."""
    parser = argparse.ArgumentParser(description="Convert raw gene expression datasets to standardized format")
    parser.add_argument('--raw-dir', default='/data/raw_data', help='Directory containing raw datasets')
    parser.add_argument('--target-dir', default='/data/gold', help='Output directory for processed datasets')
    parser.add_argument('--dataset', required=True, help="Process only a single dataset ID (e.g. GSE19615)")
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    raw_data_dir = Path(args.raw_dir)
    target_dir = Path(args.target_dir)
    
    print_now("="*80)
    print_now("DATASET CONVERSION")
    print_now("="*80)
    print_now(f"Raw data directory: {raw_data_dir}")
    print_now(f"Target directory: {target_dir}")
    
    # Scan for datasets
    datasets = scan_for_datasets(raw_data_dir)

    if args.dataset:
        # Filter to requested dataset only
        datasets = [d for d in datasets if d['id'].lower() == args.dataset.lower()]
        if not datasets:
            print_now(f"Dataset {args.dataset} not found in {raw_data_dir}")
            return 1
        
    print_now(f"\n📊 Found {len(datasets)} dataset(s) to process")
    
    if not datasets:
        print_now("❌ No valid datasets found")
        return 1
    
    # Process each dataset
    successful = 0
    failed = 0
    
    for i, dataset in enumerate(datasets, 1):
        print_now(f"\n[{i}/{len(datasets)}] Processing {dataset['id']}...")
        
        success = process_dataset(
            raw_folder_path=dataset['path'],
            dataset_id=dataset['id'],
            output_base_dir=target_dir,
            debug=args.debug
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print_now("\n" + "="*80)
    print_now("CONVERSION SUMMARY")
    print_now("="*80)
    print_now(f"✅ Successful: {successful}")
    print_now(f"❌ Failed: {failed}")
    print_now(f"📊 Success rate: {successful/(successful+failed)*100:.1f}%")
    
    # Verify processed datasets
    if successful > 0:
        verify_datasets(datasets, target_dir)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())