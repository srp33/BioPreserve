import os
import re
import sys
import traceback
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import tables
import numpy as np


def printnow(*args, **kwargs):
    """Print with immediate flush to ensure real-time output."""
    print(*args, **kwargs, flush=True)


def sanitize_column_name(col_name):
    """Convert column name to valid Python identifier for HDF5 compatibility."""
    col_name = str(col_name)
    # Replace invalid characters with underscore
    col_name = re.sub(r'[^a-zA-Z0-9_]+', '_', col_name)
    col_name = col_name.strip('_')
    
    # Add prefix if starts with digit
    if col_name and col_name[0].isdigit():
        col_name = f"col_{col_name}"
    
    return col_name or "unnamed_col"


def make_unique_columns(columns):
    """Add numerical suffixes to duplicate column names."""
    new_columns = []
    seen = set()
    
    for col in columns:
        original = col
        suffix = 1
        while col in seen:
            col = f"{original}_{suffix}"
            suffix += 1
        seen.add(col)
        new_columns.append(col)
    
    return new_columns

def process_metadata(tsv_path, h5_store):
    """
    Load metadata TSV and save to HDF5 with cleaned column names.
    This version uses a hybrid storage approach to prevent memory errors.
    """
    printnow(f"  -> Processing metadata: {tsv_path.name}")
    MAX_STRING_LENGTH = 256  # Maximum length for string columns
    
    try:
        df = pd.read_csv(tsv_path, sep='\t', low_memory=False)

        # --- Truncate excessively long strings ---
        printnow(f"    Checking for and truncating strings longer than {MAX_STRING_LENGTH} characters...")
        for col in df.select_dtypes(include=['object']).columns:
            # Find the actual max length in the column (handling potential NaN values)
            max_len = df[col].astype(str).str.len().max()
            if max_len > MAX_STRING_LENGTH:
                printnow(f"      - Truncating column '{col}' (max observed length: {int(max_len)})")
                df[col] = df[col].astype(str).str.slice(0, MAX_STRING_LENGTH)

        # Clean column names for HDF5 compatibility
        df.columns = [sanitize_column_name(col) for col in df.columns]
        df.columns = make_unique_columns(df.columns.tolist())
        
        index_col_name = df.columns[0]
        printnow(f"    Sanitizing index column '{index_col_name}' to match expression sample names...")
        # ADDED: Sanitize index values. # Pending, might fix sample ID mismatch.
        df[index_col_name] = df[index_col_name].apply(sanitize_column_name)
        
        # Use first column as dataframe index (now with sanitized values)
        df.set_index(index_col_name, inplace=True)

        printnow(f"    Found {len(df)} samples with {len(df.columns)} metadata columns.")

        # Separate columns into those that need indexing and those that don't.
        columns_to_index = [col for col in df.columns if col.startswith('refinebio') or col == 'experiment_accession_code']
        other_columns = [col for col in df.columns if col not in columns_to_index]

        # Store the indexable columns in 'table' format
        if columns_to_index:
            df_indexed = df[columns_to_index]
            h5_key_indexed = f"/metadata/{tsv_path.stem}_indexed"
            printnow(f"   {len(columns_to_index)} columns to be indexed: {columns_to_index}")
            printnow(f"    Saving indexed metadata to: {h5_key_indexed} (format=table)")
            h5_store.put(
                h5_key_indexed,
                df_indexed,
                format='table',
                data_columns=columns_to_index
            )
            printnow(f"    ✓ Saved: {h5_key_indexed}")

        # # Store the remaining (majority of) columns in 'fixed' format to save memory
        # if other_columns:
        #     df_other = df[other_columns]
        #     h5_key_other = f"/metadata/{tsv_path.stem}_other_data"
        #     printnow(f"    Saving {len(other_columns)} non-indexed metadata columns to: {h5_key_other} (format=fixed)")
        #     h5_store.put(
        #         h5_key_other,
        #         df_other,
        #         format='fixed'
        #     )
        #     printnow(f"    ✓ Saved: {h5_key_other}")
        
    except Exception as e:
        printnow(f"    ✗ Error processing metadata file {tsv_path.name}: {e}", file=sys.stderr)
        traceback.print_exc()


def process_expression(tsv_path, h5_store):
    """
    Process large expression matrix (genes x samples) using chunked reading.
    Stores data in genes x samples orientation without transposing.
    """
    printnow(f"  -> Processing expression file (no transpose): {tsv_path.name}")

    try:
        printnow("    Reading header to get sample names...")
        with open(tsv_path, 'r') as f:
            # Remove trailing tabs and newlines
            header_line = f.readline().strip() 
        
        sanitized_header = re.sub(r'[^a-zA-Z0-9_\t]+', '_', header_line)
        sample_names = sanitized_header.split('\t')
        sample_names = make_unique_columns(sample_names)
        
        sanitized_sample_names = [sanitize_column_name(s) for s in sample_names]
        sanitized_sample_names = make_unique_columns(sanitized_sample_names)

        num_samples = len(sanitized_sample_names)
        printnow(f"    Found {num_samples} samples.")

        print("Creating HDF5 structure...")
        h5_file = h5_store.root._v_file
        base_key = f"/expression/{tsv_path.stem}"
        parent, group = ('/' + base_key.strip('/')).rsplit('/', 1)
        h5_group = h5_file.create_group(parent, group, createparents=True)

        printnow(f"    Saving sample names as fixed array to: {base_key}/sample_names")
        h5_file.create_array(h5_group, 'sample_names', obj=np.array(sanitized_sample_names, dtype='S'))

        printnow(f"    Creating extendable arrays for gene names")
        gene_names_earray = h5_file.create_earray(
            h5_group,
            'gene_names',
            atom=tables.StringAtom(itemsize=256),
            shape=(0,),
            title='Gene Names'
        )

        printnow(f"    Creating extendable array for expression data")
        expression_earray = h5_file.create_earray(
            h5_group,
            'expression_data',
            atom=tables.Float32Atom(),
            shape=(0, num_samples),  # genes x samples
            title='Expression Data (Genes x Samples)',
            filters=tables.Filters(complevel=5, complib='zlib')
        )

        # Process file in chunks to handle large files
        chunk_size = 500
        printnow(f"    Processing file in chunks of {chunk_size}...")
        processed_genes = 0
        
        # We tell pandas to use our column names to be extra safe
        reader = pd.read_csv(tsv_path, sep='\t', chunksize=chunk_size, index_col=0, low_memory=False,
                             header=0, names=sample_names)
        
        for i, chunk in enumerate(reader):
            if chunk.empty:
                continue

            # This debugging check should now pass for all chunks!
            if chunk.shape[1] != num_samples:
                printnow(f"    ✗ ERROR: Shape mismatch found in chunk {i}! "
                         f"Expected {num_samples}, got {chunk.shape[1]}", file=sys.stderr)
                raise ValueError("Stopping due to column mismatch.")

            # Store gene names and expression values
            gene_names_earray.append(chunk.index.values.astype('S'))
            expression_data = chunk.values.astype(np.float32)
            expression_earray.append(expression_data)
            
            processed_genes += len(chunk)
            if (i + 1) % 5 == 0:
                printnow(f"    ...processed {processed_genes} genes")

        printnow(f"    ✓ Successfully saved data for {processed_genes} genes to: {base_key}")

    except Exception as e:
        printnow(f"    ✗ Error during expression file processing: {e}", file=sys.stderr)
        traceback.print_exc()


def create_transposed_view(h5_path, base_key):
    """
    Create memory-efficient transposed copy of expression data (samples x genes).
    Processes data in chunks to avoid loading entire matrix into memory.
    """
    printnow(f"\n  -> Creating transposed view for {base_key}...")
    try:
        with tables.open_file(str(h5_path), mode='a', filters=tables.Filters(complevel=5, complib='zlib')) as h5_file:
            source_group = h5_file.get_node(base_key)
            source_data = source_group.expression_data

            # Skip if transposed data already exists
            if 'expression_data_transposed' in source_group:
                printnow("    - Transposed view already exists. Skipping.")
                return

            num_genes, num_samples = source_data.shape
            printnow(f"    - Original matrix shape (Genes x Samples): {num_genes} x {num_samples}")

            # Create compressed array for transposed data
            transposed_matrix = h5_file.create_carray(
                source_group,
                'expression_data_transposed',
                atom=tables.Float32Atom(),
                shape=(num_samples, num_genes),
                title="Expression Data Transposed (Samples x Genes)"
            )

            # Process in chunks to manage memory usage
            chunk_size = 500
            for i in range(0, num_genes, chunk_size):
                chunk_end = min(i + chunk_size, num_genes)
                gene_chunk = source_data[i:chunk_end, :]
                transposed_matrix[:, i:chunk_end] = gene_chunk.T
                if (i // chunk_size + 1) % 5 == 0:
                     printnow(f"    ...transposed {chunk_end}/{num_genes} genes")
            
            printnow(f"    ✓ Successfully created transposed view.")

    except Exception as e:
        printnow(f"    ✗ Error creating transposed view for {base_key}: {e}", file=sys.stderr)
        traceback.print_exc()


def is_metadata_file(filename):
    return 'metadata' in filename.lower()


def main():
    """
    Convert TSV files to organized HDF5 format.
    Separates metadata files from expression data based on filename patterns.
    """
    parser = ArgumentParser(description="Convert TSV files to organized HDF5 format.")
    parser.add_argument("-i", "--input-directory", type=Path, required=True,
                       help="Directory containing TSV files")
    parser.add_argument("-o", "--output-path", type=Path, required=True,
                       help="Output HDF5 file path")
    parser.add_argument("--transpose", action="store_true",
                       help="Also create a transposed (samples x genes) view of expression data.")
    
    args = parser.parse_args()
    
    if not args.input_directory.is_dir():
        print(f"Error: Directory not found: {args.input_directory}", file=sys.stderr)
        sys.exit(1)
        
    printnow(f"Starting conversion process.")
    printnow(f"Input Directory: {args.input_directory}")
    printnow(f"Output HDF5 File: {args.output_path}")
    
    processed_expression_keys = []

    # Process all TSV files in directory
    with pd.HDFStore(args.output_path, 'w', complevel=5, complib='zlib') as h5_store:
        # Sort for consistent processing order
        for dirpath, dirnames, filenames in os.walk(args.input_directory, topdown=True):
            dirnames.sort()
            
            for filename in sorted(filenames):
                if not filename.lower().endswith('.tsv'):
                    continue

                tsv_path = Path(dirpath) / filename
                relative_str = str(tsv_path.relative_to(args.input_directory))
                relative_path = relative_str.replace('\\', '/')
                printnow(f"\nProcessing: {relative_path}")

                # if relative_path in ["HOMO_SAPIENS.tsv", "filtered_samples_metadata.tsv", "HOMO_SAPIENS/metadata_HOMO_SAPIENS.tsv"]:
                #     printnow(f"    - Skipping file: {relative_path}")
                #     continue

                
                # Route to appropriate processor based on filename
                if is_metadata_file(filename):
                    process_metadata(tsv_path, h5_store)
                else:
                    process_expression(tsv_path, h5_store)
                    processed_expression_keys.append(f"/expression/{tsv_path.stem}")
    
    printnow(f"\nInitial data conversion complete. Data saved to: {args.output_path}")

    # Create transposed views if requested
    if args.transpose:
        printnow("\nCreating transposed views as requested...")
        for base_key in processed_expression_keys:
            create_transposed_view(args.output_path, base_key)
        printnow("\nTransposing complete.")
    else:
        printnow("\nSkipping transpose step. Use the --transpose flag to enable it.")


if __name__ == "__main__":
    main()