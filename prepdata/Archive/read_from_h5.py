import pandas as pd
import tables

def read_expression_data(h5_path, dataset_name):
    """
    Reads the custom-stored expression data from the HDF5 file and
    reconstructs it into a pandas DataFrame.
    """
    base_key = f"/expression/{dataset_name}"
    
    with pd.HDFStore(h5_path, 'r') as h5_store:
        # Access the low-level file object
        h5_file = h5_store.get_node(base_key)._v_file
        
        # Read the three separate components
        expression_data = h5_file.get_node(base_key, 'expression_data').read()
        gene_names = h5_file.get_node(base_key, 'gene_names').read().astype(str)
        sample_ids = h5_file.get_node(base_key, 'sample_ids').read().astype(str)
        
        # Reconstruct the DataFrame
        df = pd.DataFrame(expression_data, index=sample_ids, columns=gene_names)
        
        return df

# Example usage:
# df = read_expression_data('your_output_file.h5', 'HOMO_SAPIENS')
# print(df.head())