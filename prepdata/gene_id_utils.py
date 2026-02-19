#!/usr/bin/env python3
"""
Gene ID Utilities Module

This module provides functions for detecting gene ID types and converting them to gene symbols.
It can be used both as a standalone analysis tool and as an imported module.

Supported Gene ID Types:
- Ensembl Gene IDs (ENSG...) - via BioMart
- Entrez Gene IDs (numeric) - via BioMart  
- RefSeq IDs (NM_..., NR_...) - via BioMart
- Gene Symbols (already symbols, no conversion needed)
- Affymetrix Probe IDs (requires platform-specific files)

BioMart Integration:
This module uses Ensembl's BioMart service to automatically create gene ID mappings.
BioMart provides access to many gene identifier types including:
- ensembl_gene_id: Ensembl gene identifiers
- entrezgene_id: NCBI Entrez gene identifiers  
- refseq_mrna: RefSeq mRNA identifiers
- refseq_ncrna: RefSeq non-coding RNA identifiers
- external_gene_name: HGNC gene symbols
- And many more...

For a full list of available attributes, you can query:
```python
from pybiomart import Dataset
dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
attributes = dataset.attributes
print(attributes[attributes['name'].str.contains('gene|symbol|entrez|refseq')])
```
"""

import pandas as pd
import re
from pathlib import Path
import argparse

def print_now(*args, **kwargs):
    """Prints a message to the console with flushing to ensure immediate output."""
    print(*args, flush=True, **kwargs)

def detect_gene_id_type(gene_ids, debug=False):
    """
    Detect the type of gene identifiers from a list of gene IDs.
    
    Args:
        gene_ids (list): List of gene identifiers
        debug (bool): Enable debug output
    
    Returns:
        dict: Detection results with type, confidence, and examples
    """
    if not gene_ids:
        return {'type': 'unknown', 'confidence': 0.0, 'examples': []}
    
    # Convert to strings and take a sample
    gene_ids_str = [str(gid) for gid in gene_ids[:1000]]  # Sample first 1000
    total_count = len(gene_ids_str)
    
    # Detection patterns
    patterns = {
        'ensembl': {
            'pattern': r'^ENSG\d{11}',
            'description': 'ENSEMBL Gene IDs (ENSG...)'
        },
        'entrez': {
            'pattern': r'^\d+$',
            'description': 'Entrez Gene IDs (numeric)'
        },
        'probe_affymetrix': {
            'pattern': r'^\d+_[a-z]?_?at$',
            'description': 'Affymetrix Probe IDs (..._at)'
        },
        'gene_symbol': {
            'pattern': r'^[A-Z][A-Z0-9-]*$',
            'description': 'Gene Symbols (e.g., TP53, BRCA1)'
        },
        'refseq': {
            'pattern': r'^N[MR]_\d+',
            'description': 'RefSeq IDs (NM_... or NR_...)'
        }
    }
    
    results = {}
    
    for id_type, pattern_info in patterns.items():
        pattern = pattern_info['pattern']
        matches = sum(1 for gid in gene_ids_str if re.match(pattern, gid, re.IGNORECASE))
        confidence = matches / total_count if total_count > 0 else 0.0
        
        results[id_type] = {
            'matches': matches,
            'confidence': confidence,
            'description': pattern_info['description']
        }
    
    # Find the best match
    best_type = max(results.keys(), key=lambda k: results[k]['confidence'])
    best_confidence = results[best_type]['confidence']
    print_now(f"   {best_type}: {results[best_type]['matches']}/{total_count} matches ({best_confidence:.1%})")
    
    # Get examples
    examples = gene_ids_str[:5]
    
    return {
        'type': best_type if best_confidence > 0.5 else 'unknown',
        'confidence': best_confidence,
        'examples': examples,
        'all_results': results
    }

def create_gene_mapping(gene_id_type, annotation_dir="grp_batch_effects/data/annotations", verbose=True):
    """
    Create or load gene ID to symbol mapping based on detected type.
    
    Args:
        gene_id_type (str): Detected gene ID type
        annotation_dir (str): Directory for annotation files
        verbose (bool): Enable verbose output
    
    Returns:
        dict: Mapping from gene IDs to gene symbols
    """
    annotation_path = Path(annotation_dir)
    annotation_path.mkdir(parents=True, exist_ok=True)
    
    if gene_id_type == 'ensembl':
        annotation_file = annotation_path / 'ensembl_to_symbol_map.csv'
        
        # Create mapping if it doesn't exist
        if not annotation_file.exists():
            if verbose:
                print_now(f"   🔄 Creating Ensembl mapping from BioMart...")
            try:
                from pybiomart import Dataset
                
                # Connect to Ensembl BioMart (human genes)
                dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
                
                # Query ensembl_gene_id + gene name
                annot = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
                annot.columns = ['EnsemblID', 'GeneSymbol']
                
                # Save to CSV
                annot.to_csv(annotation_file, index=False)
                if verbose:
                    print_now(f"   ✅ Created {annotation_file} with {len(annot)} mappings")
                
            except ImportError:
                if verbose:
                    print_now("   ❌ pybiomart not available. Install with: pip install pybiomart")
                return {}
            except Exception as e:
                if verbose:
                    print_now(f"   ❌ Failed to create Ensembl mapping: {e}")
                return {}
        
        # Load the mapping
        try:
            annot = pd.read_csv(annotation_file)
            annot = annot.dropna(subset=['EnsemblID', 'GeneSymbol'])
            annot = annot.drop_duplicates(subset=['EnsemblID'], keep='first')
            return annot.set_index('EnsemblID')['GeneSymbol'].to_dict()
        except Exception as e:
            if verbose:
                print_now(f"   ❌ Failed to load Ensembl mapping: {e}")
            return {}
    
    elif gene_id_type == 'entrez':
        annotation_file = annotation_path / 'entrez_to_symbol_map.csv'
        
        # Create mapping if it doesn't exist
        if not annotation_file.exists():
            if verbose:
                print_now(f"   🔄 Creating Entrez mapping from BioMart...")
            try:
                from pybiomart import Dataset
                
                # Connect to Ensembl BioMart (human genes)
                dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
                
                # Query entrez_gene_id + gene name
                annot = dataset.query(attributes=['entrezgene_id', 'external_gene_name'])
                annot.columns = ['EntrezID', 'GeneSymbol']
                
                # Remove rows with missing Entrez IDs
                annot = annot.dropna(subset=['EntrezID'])
                annot['EntrezID'] = annot['EntrezID'].astype(int).astype(str)  # Convert to string for consistency
                
                # Save to CSV
                annot.to_csv(annotation_file, index=False)
                if verbose:
                    print_now(f"   ✅ Created {annotation_file} with {len(annot)} mappings")
                
            except ImportError:
                if verbose:
                    print_now("   ❌ pybiomart not available. Install with: pip install pybiomart")
                return {}
            except Exception as e:
                if verbose:
                    print_now(f"   ❌ Failed to create Entrez mapping: {e}")
                return {}
        
        # Load the mapping
        try:
            annot = pd.read_csv(annotation_file)
            annot = annot.dropna(subset=['EntrezID', 'GeneSymbol'])
            annot = annot.drop_duplicates(subset=['EntrezID'], keep='first')
            # Convert EntrezID to string for mapping
            annot['EntrezID'] = annot['EntrezID'].astype(str)
            return annot.set_index('EntrezID')['GeneSymbol'].to_dict()
        except Exception as e:
            if verbose:
                print_now(f"   ❌ Failed to load Entrez mapping: {e}")
            return {}
    
    elif gene_id_type == 'refseq':
        annotation_file = annotation_path / 'refseq_to_symbol_map.csv'
        
        # Create mapping if it doesn't exist
        if not annotation_file.exists():
            if verbose:
                print_now(f"   🔄 Creating RefSeq mapping from BioMart...")
            try:
                from pybiomart import Dataset
                
                # Connect to Ensembl BioMart (human genes)
                dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
                
                # Query RefSeq mRNA and ncRNA IDs + gene name
                annot = dataset.query(attributes=['refseq_mrna', 'refseq_ncrna', 'external_gene_name'])
                
                # Combine RefSeq mRNA and ncRNA into one column
                refseq_combined = []
                gene_symbols = []
                
                for _, row in annot.iterrows():
                    gene_symbol = row['external_gene_name']
                    if pd.notna(row['refseq_mrna']) and row['refseq_mrna']:
                        refseq_combined.append(row['refseq_mrna'])
                        gene_symbols.append(gene_symbol)
                    if pd.notna(row['refseq_ncrna']) and row['refseq_ncrna']:
                        refseq_combined.append(row['refseq_ncrna'])
                        gene_symbols.append(gene_symbol)
                
                # Create final dataframe
                final_annot = pd.DataFrame({
                    'RefSeqID': refseq_combined,
                    'GeneSymbol': gene_symbols
                })
                
                # Remove rows with missing data
                final_annot = final_annot.dropna(subset=['RefSeqID', 'GeneSymbol'])
                final_annot = final_annot[final_annot['RefSeqID'] != '']
                
                # Save to CSV
                final_annot.to_csv(annotation_file, index=False)
                if verbose:
                    print_now(f"   ✅ Created {annotation_file} with {len(final_annot)} mappings")
                
            except ImportError:
                if verbose:
                    print_now("   ❌ pybiomart not available. Install with: pip install pybiomart")
                return {}
            except Exception as e:
                if verbose:
                    print_now(f"   ❌ Failed to create RefSeq mapping: {e}")
                return {}
        
        # Load the mapping
        try:
            annot = pd.read_csv(annotation_file)
            annot = annot.dropna(subset=['RefSeqID', 'GeneSymbol'])
            annot = annot.drop_duplicates(subset=['RefSeqID'], keep='first')
            return annot.set_index('RefSeqID')['GeneSymbol'].to_dict()
        except Exception as e:
            if verbose:
                print_now(f"   ❌ Failed to load RefSeq mapping: {e}")
            return {}
    
    elif gene_id_type == 'probe_affymetrix':
        # Affymetrix probe mapping would require platform-specific annotation files
        # This is more complex as it depends on the specific microarray platform
        if verbose:
            print_now(f"   ⚠️  Affymetrix probe mapping requires platform-specific annotation files")
        return {}
    
    elif gene_id_type == 'gene_symbol':
        if verbose:
            print_now(f"   ✅ Already gene symbols - no mapping needed")
        return {}
    
    else:
        if verbose:
            print_now(f"   ⚠️  Unknown gene ID type: {gene_id_type}")
        return {}

def convert_gene_ids_to_symbols(expr_df, gene_id_type, annotation_dir="grp_batch_effects/data/annotations", debug=False):
    """
    Convert gene IDs to gene symbols in expression dataframe.
    
    Args:
        expr_df (pd.DataFrame): Expression dataframe with gene IDs as columns
        gene_id_type (str): Type of gene IDs
        annotation_dir (str): Directory for annotation files
        debug (bool): Enable debug output
    
    Returns:
        pd.DataFrame: Expression dataframe with gene symbols as columns
    """
    if gene_id_type == 'gene_symbol':
        if debug:
            print_now(f"   ✅ Gene IDs are already symbols")
        return expr_df
    
    if debug:
        print_now(f"   🔄 Converting {gene_id_type} IDs to gene symbols...")
    
    # Get the mapping
    gene_mapping = create_gene_mapping(gene_id_type, annotation_dir, verbose=debug)
    
    if not gene_mapping:
        if debug:
            print_now(f"   ⚠️  No mapping available, keeping original IDs")
        return expr_df
    
    # Separate metadata from expression data
    meta_cols = [col for col in expr_df.columns if col.startswith('meta_')]
    meta_df = expr_df[meta_cols]
    gene_cols = [col for col in expr_df.columns if not col.startswith('meta_')]
    gene_df = expr_df[gene_cols]
    
    # Transpose, map gene IDs to symbols, and aggregate by taking the mean
    gene_df_T = gene_df.T
    gene_df_T['GeneSymbol'] = gene_df_T.index.map(gene_mapping)
    
    if debug:
        mapped_count = gene_df_T['GeneSymbol'].notna().sum()
        total_count = len(gene_df_T)
        print_now(f"   📊 Mapped {mapped_count}/{total_count} {gene_id_type} IDs to gene symbols")
    
    # Drop unmapped genes and aggregate by gene symbol
    gene_df_T = gene_df_T.dropna(subset=['GeneSymbol'])
    gene_df_T_grouped = gene_df_T.groupby('GeneSymbol').mean(numeric_only=True)
    
    # Transpose back and combine with metadata
    gene_df_final = gene_df_T_grouped.T
    final_df = pd.concat([meta_df.reset_index(drop=True), gene_df_final.reset_index(drop=True)], axis=1)
    
    if debug:
        print_now(f"   ✅ Gene ID conversion complete. Shape: {expr_df.shape} → {final_df.shape}")
    return final_df

def suggest_annotation_file(gene_id_type, annotation_dir="grp_batch_effects/data/annotations"):
    """
    Suggest appropriate annotation file based on detected gene ID type.
    
    Args:
        gene_id_type (str): Detected gene ID type
        annotation_dir (str): Directory containing annotation files
    
    Returns:
        dict: Suggestion with file path and mapping type
    """
    annotation_path = Path(annotation_dir)
    
    suggestions = {
        'ensembl': {
            'file': 'ensembl_to_symbol_map.csv',
            'map_type': 'ensembl',
            'description': 'ENSEMBL ID to Gene Symbol mapping'
        },
        'entrez': {
            'file': 'entrez_to_symbol_map.csv',
            'map_type': 'entrez',
            'description': 'Entrez ID to Gene Symbol mapping'
        },
        'refseq': {
            'file': 'refseq_to_symbol_map.csv',
            'map_type': 'refseq',
            'description': 'RefSeq ID to Gene Symbol mapping'
        },
        'probe_affymetrix': {
            'file': 'GPL96-annotation.csv',
            'map_type': 'probe',
            'description': 'Affymetrix Probe ID to Gene Symbol mapping'
        },
        'gene_symbol': {
            'file': None,
            'map_type': 'none',
            'description': 'Already gene symbols - no mapping needed'
        }
    }
    
    if gene_id_type not in suggestions:
        return {
            'file': None,
            'map_type': 'none',
            'description': f'Unknown gene ID type: {gene_id_type}',
            'exists': False
        }
    
    suggestion = suggestions[gene_id_type].copy()
    
    if suggestion['file']:
        file_path = annotation_path / suggestion['file']
        suggestion['full_path'] = str(file_path)
        suggestion['exists'] = file_path.exists()
    else:
        suggestion['full_path'] = None
        suggestion['exists'] = True  # No file needed
    
    return suggestion

def analyze_dataset(file_path, debug=False):
    """
    Analyze a dataset file to detect gene ID types and suggest mappings.
    
    Args:
        file_path (str): Path to the dataset CSV file
        debug (bool): Enable debug output
    
    Returns:
        dict: Analysis results
    """
    print_now(f"🔍 Analyzing dataset: {file_path}")
    
    try:
        # Read the dataset
        df = pd.read_csv(file_path, low_memory=False)
        
        # Separate metadata from gene columns
        meta_cols = [col for col in df.columns if col.startswith('meta_')]
        gene_cols = [col for col in df.columns if not col.startswith('meta_') and col != 'Sample_ID']
        
        print_now(f"   📊 Dataset shape: {df.shape}")
        print_now(f"   📊 Gene columns: {len(gene_cols)}")
        print_now(f"   📊 Metadata columns: {len(meta_cols)}")
        
        if not gene_cols:
            return {
                'file_path': file_path,
                'error': 'No gene columns found',
                'gene_id_detection': None,
                'suggestion': None
            }
        
        # Detect gene ID type
        detection = detect_gene_id_type(gene_cols, debug=debug)
        print_now(f"   🎯 Detected gene ID type: {detection['type']} (confidence: {detection['confidence']:.1%})")
        print_now(f"   📝 Examples: {', '.join(detection['examples'])}")
        
        # Get suggestion
        suggestion = suggest_annotation_file(detection['type'])
        
        if suggestion['file']:
            status = "✅ Available" if suggestion['exists'] else "❌ Missing"
            print_now(f"   💡 Suggested mapping: {suggestion['description']}")
            print_now(f"   📁 Annotation file: {suggestion['file']} ({status})")
        else:
            print_now(f"   💡 {suggestion['description']}")
        
        return {
            'file_path': file_path,
            'dataset_shape': df.shape,
            'gene_columns': len(gene_cols),
            'meta_columns': len(meta_cols),
            'gene_id_detection': detection,
            'suggestion': suggestion,
            'error': None
        }
        
    except Exception as e:
        print_now(f"   ❌ Error analyzing dataset: {e}")
        return {
            'file_path': file_path,
            'error': str(e),
            'gene_id_detection': None,
            'suggestion': None
        }

def main():
    """Main function for standalone usage - analyzes datasets and reports gene ID types."""
    parser = argparse.ArgumentParser(description="Detect gene ID types in datasets and suggest annotation files")
    parser.add_argument('files', nargs='+', help='Dataset files to analyze')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--annotation-dir', default='grp_batch_effects/data/annotations',
                        help='Directory containing annotation files')
    
    args = parser.parse_args()
    
    print_now("="*80)
    print_now("GENE ID TYPE DETECTION")
    print_now("="*80)
    
    results = []
    
    for file_path in args.files:
        result = analyze_dataset(file_path, debug=args.debug)
        results.append(result)
        print_now()  # Empty line between files
    
    # Summary
    print_now("="*80)
    print_now("ANALYSIS SUMMARY")
    print_now("="*80)
    
    type_counts = {}
    missing_annotations = []
    
    for result in results:
        if result['error']:
            print_now(f"❌ {result['file_path']}: {result['error']}")
            continue
            
        detection = result['gene_id_detection']
        suggestion = result['suggestion']
        
        gene_type = detection['type']
        type_counts[gene_type] = type_counts.get(gene_type, 0) + 1
        
        status = "✅" if suggestion['exists'] else "❌"
        print_now(f"{status} {result['file_path']}: {gene_type} ({detection['confidence']:.1%})")
        
        if suggestion['file'] and not suggestion['exists']:
            missing_annotations.append(suggestion['file'])
    
    print_now(f"\n📊 Gene ID Type Distribution:")
    for gene_type, count in type_counts.items():
        print_now(f"   {gene_type}: {count} datasets")
    
    if missing_annotations:
        print_now(f"\n⚠️  Missing annotation files:")
        for file in set(missing_annotations):
            print_now(f"   - {file}")
        print_now(f"\nNote: Gene ID conversion is now handled automatically in convert_raw_files.py")

if __name__ == "__main__":
    main()