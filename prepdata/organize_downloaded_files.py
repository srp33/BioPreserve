"""
File organizer for moving and categorizing downloaded files.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional

from analyzers.content_analyzer import ContentAnalyzer, AnalysisResult
from config import FileType
from utils import FileSystemError


import argparse


class FileOrganizer:
    """Organizer for moving files from raw_download to raw_data with proper categorization."""
    
    def __init__(self, analyzer: ContentAnalyzer):
        """Initialize with content analyzer."""
        self.analyzer = analyzer
    
    def organize_dataset(self, dataset_id: str, raw_dir: Path, target_dir: Path) -> bool:
        """Organize all files for a specific dataset."""
        try:
            print(f"📁 Organizing dataset: {dataset_id}")
            
            # Find dataset directory in raw downloads
            dataset_raw_dir = raw_dir / dataset_id.lower()
            if not dataset_raw_dir.exists():
                print(f"⚠️  Dataset directory not found: {dataset_raw_dir}")
                return False
            
            # Create target dataset directory with exact naming to match existing structure
            # Directory names should be lowercase versions of dataset IDs
            target_dir_name = self._normalize_directory_name(dataset_id)
            dataset_target_dir = target_dir / target_dir_name
            dataset_target_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all files in the dataset directory
            files_processed = 0
            files_success = 0
            
            for file_path in dataset_raw_dir.iterdir():
                if file_path.is_file():
                    files_processed += 1
                    target_path = self.categorize_and_move(file_path, dataset_target_dir, dataset_id)
                    if target_path:
                        files_success += 1
                        print(f"✅ Organized: {file_path.name} → {target_path.name}")
            
            print(f"📊 Organized {files_success}/{files_processed} files for {dataset_id}")
            return files_success > 0
            
        except Exception as e:
            print(f"❌ Error organizing dataset {dataset_id}: {e}")
            return False
    
    def categorize_and_move(self, file_path: Path, target_dir: Path, dataset_id: str) -> Optional[Path]:
        """Analyze file, determine category, and move to appropriate location."""
        try:
            # Analyze file content
            analysis = self.analyzer.analyze_file(file_path)
            
            # Determine target filename based on analysis
            target_name = self._generate_target_name(analysis, dataset_id, file_path)
            if not target_name:
                print(f"Skipping move for {file_path}.")
                return None
            target_path = target_dir / target_name
            
            # Copy file to target location
            shutil.copy2(str(file_path), str(target_path))
            
            return target_path
            
        except Exception as e:
            print(f"❌ Error categorizing file {file_path}: {e}")
            return None
    
    def organize_all_datasets(self, raw_dir: Path, target_dir: Path) -> Dict[str, bool]:
        """Organize all datasets found in raw download directory."""
        results = {}
        
        if not raw_dir.exists():
            print(f"⚠️  Raw download directory not found: {raw_dir}")
            return results
        
        # Find all dataset directories
        dataset_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        
        if not dataset_dirs:
            print("⚠️  No dataset directories found in raw downloads")
            return results
        
        print(f"📁 Found {len(dataset_dirs)} datasets to organize")
        
        for dataset_dir in dataset_dirs:
            dataset_id = dataset_dir.name.upper()  # Normalize to uppercase
            success = self.organize_dataset(dataset_id, raw_dir, target_dir)
            results[dataset_id] = success
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        print(f"📊 Organization complete: {successful}/{len(results)} datasets organized successfully")
        
        return results
    
    def _generate_target_name(self, analysis: AnalysisResult, dataset_id: str, original_path: Path) -> str:
        """Generate target filename based on analysis results to match existing structure exactly."""
        
        # Preserve original file extension to maintain compatibility
        extension = original_path.suffix
        if not extension:
            # Default based on content type to match existing structure
            if analysis.content_type == FileType.METADATA:
                extension = '.tsv'  # Metadata files are typically .tsv
            else:
                extension = '.csv'  # Expression files can be .csv
        
        # Normalize dataset_id to match existing naming patterns exactly
        # The dataset_id should already be in the correct format (e.g., GSE62944_Tumor)
        normalized_dataset_id = dataset_id
        
        # Generate name based on content type with exact naming convention
        if analysis.content_type == FileType.EXPRESSION:
            return f"expression_{normalized_dataset_id}{extension}"
        
        elif analysis.content_type == FileType.METADATA:
            return f"meta_{normalized_dataset_id}{extension}"

        elif analysis.content_type == FileType.NON_TABULAR:
            # Skip non-tabular data
            return None
        
        else:
            # Unknown type, use filename-based heuristics to match existing patterns
            original_lower = original_path.stem.lower()
            
            # Check for expression indicators
            if any(keyword in original_lower for keyword in ['expression', 'expr', 'gene', 'probe', 'data', 'matrix']):
                return f"expression_{normalized_dataset_id}{extension}"
            # Check for metadata indicators  
            elif any(keyword in original_lower for keyword in ['meta', 'clinical', 'sample', 'patient', 'pheno']):
                return f"meta_{normalized_dataset_id}{extension}"
            # Check file extension as fallback
            elif extension.lower() == '.csv':
                # CSV files are more likely to be expression data
                return f"expression_{normalized_dataset_id}{extension}"
            elif extension.lower() == '.tsv':
                # TSV files could be either, but default to metadata
                return f"meta_{normalized_dataset_id}{extension}"
            else:
                # Final fallback - default to expression
                return f"expression_{normalized_dataset_id}{extension}"
    
    def _normalize_directory_name(self, dataset_id: str) -> str:
        """Normalize dataset ID to directory name to match existing structure exactly."""
        # Convert to lowercase and handle special cases to match existing structure
        normalized = dataset_id.lower()
        
        # Handle specific special cases observed in existing structure
        special_cases = {
            'gse62944_tumor': 'gse62944_tumor',
            'gse96058_hiseq': 'gse96058_hiseq', 
            'gse96058_nextseq': 'gse96058_nextseq',
            'metabric': 'metabric'
        }
        
        # Check if this is a special case
        if normalized in special_cases:
            return special_cases[normalized]
        
        # For regular GSE datasets, just use lowercase
        return normalized
    
    
    def verify_organization(self, target_dir: Path, dataset_ids: List[str]) -> Dict[str, Dict[str, bool]]:
        """Verify that datasets were organized correctly."""
        results = {}
        
        for dataset_id in dataset_ids:
            dataset_dir = target_dir / dataset_id.lower()
            
            # Check for expression and metadata files
            expression_files = list(dataset_dir.glob(f"expression_{dataset_id}*"))
            metadata_files = list(dataset_dir.glob(f"meta_{dataset_id}*"))
            
            results[dataset_id] = {
                'has_expression': len(expression_files) > 0,
                'has_metadata': len(metadata_files) > 0,
                'expression_count': len(expression_files),
                'metadata_count': len(metadata_files)
            }
        
        return results


def print_verification_results(verification_results: Dict[str, Dict[str, bool]]):
    """Prints verification results in a formatted and readable table."""
    print("\n" + "="*60)
    print("📊 Verification Results")
    print("="*60)
    
    if not verification_results:
        print("No datasets were verified.")
        print("="*60)
        return
        
    # Table Header
    print(f"{'Dataset':<30} | {'Expression Files':<20} | {'Metadata Files':<20}")
    print("-" * 60)
    
    # Table Rows
    for dataset_id, results in verification_results.items():
        exp_status = f"✅ Found ({results['expression_count']})" if results['has_expression'] else "❌ Missing (0)"
        meta_status = f"✅ Found ({results['metadata_count']})" if results['has_metadata'] else "❌ Missing (0)"
        
        print(f"{dataset_id:<30} | {exp_status:<20} | {meta_status:<20}")
        
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Organize downloaded files.')
    parser.add_argument('--raw-dir', type=Path, required=True, help='Path to the raw downloads directory.')
    parser.add_argument('--target-dir', type=Path, required=True, help='Path to the target raw data directory.')

    args = parser.parse_args()

    analyzer = ContentAnalyzer(debug=True)
    organizer = FileOrganizer(analyzer)

    results = organizer.organize_all_datasets(args.raw_dir, args.target_dir)

    verification = organizer.verify_organization(args.target_dir, list(results.keys()))

    print_verification_results(verification)