#!/usr/bin/env python3
"""
Enhanced Dataset Combination Generation Script

This script finds all processed datasets containing 'meta_er_status', creates all
possible pairwise combinations for multiple CSV file types (unadjusted, combat, limma, etc.),
and saves them to the /data/paired_datasets/ directory with caching and performance
optimizations.

Key Features:
- Multi-file support: Processes all available CSV file types per dataset
- Caching: Uses HashCache to avoid regenerating unchanged combinations  
- Performance optimization: Supports parallel processing
- Comprehensive reporting: Detailed statistics on cache efficiency and performance

Usage:
    # Basic usage
    python scripts/prepdata/generate_all_combinations.py
    
    # With performance optimizations
    python scripts/prepdata/generate_all_combinations.py --parallel 4 --debug
    
    # Process specific file types only
    python scripts/prepdata/generate_all_combinations.py --csv-files unadjusted.csv combat.csv
    
    # Dry run to preview what would be generated
    python scripts/prepdata/generate_all_combinations.py --dry-run

Arguments:
    --debug                Enable detailed debug output
    --dry-run              Show what would be done without actually doing it
    --parallel N           Number of parallel processes (default: 1)
    --output-dir PATH      Output directory for combined datasets (default: /data/paired_datasets)
    --data-dir PATH        Input directory containing processed datasets (default: /data/gold)
    --cache-dir PATH       Directory for cache files (default: /tmp/combination_cache)
    --max-combinations N   Maximum number of combinations to process (for testing)
    --csv-files FILES      Specific CSV files to process (default: all available)

"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from itertools import combinations
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import time
# Optional import for parallel processing
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    CONCURRENT_FUTURES_AVAILABLE = True
except ImportError:
    CONCURRENT_FUTURES_AVAILABLE = False

import psutil

# Add the parent directory (scripts) to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



@dataclass
class DatasetInfo:
    """Information about a dataset and its available files."""
    name: str
    path: Path
    available_files: List[str]
    has_meta_er_status: bool
    sample_count: int = 0
    gene_count: int = 0

@dataclass
class CombinationResult:
    """Result of a combination operation."""
    combo_name: str
    dataset1: str
    dataset2: str
    file_type: str
    success: bool
    output_path: Path
    file_size: int = 0
    error_message: Optional[str] = None



@dataclass
class PerformanceStats:
    """Performance statistics for monitoring."""
    start_time: float = 0.0
    end_time: float = 0.0

    total_files_processed: int = 0
    total_size_mb: float = 0.0
    
    @property
    def elapsed_time(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else time.time() - self.start_time
    
    @property
    def throughput_mb_per_sec(self) -> float:
        elapsed = self.elapsed_time
        return self.total_size_mb / elapsed if elapsed > 0 else 0.0

def print_now(*args, **kwargs):
    """Prints a message to the console with flushing to ensure immediate output."""
    print(*args, flush=True, **kwargs)



def check_disk_space(path: Path, required_mb: float = 1000) -> bool:
    """Check if there's enough disk space available."""
    try:
        stat = os.statvfs(path)
        available_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        return available_mb > required_mb
    except:
        return True  # Assume OK if we can't check

def discover_csv_files(dataset_path: Path) -> List[str]:
    """
    Discover all available CSV files in a dataset directory.
    
    Returns a list of CSV filenames (without path) that exist in the dataset directory.
    """
    csv_files = []
    if dataset_path.exists() and dataset_path.is_dir():
        for file_path in dataset_path.glob("*.csv"):
            # Skip temporary and metadata files
            if not file_path.name.startswith('.') and not file_path.name.startswith('_'):
                csv_files.append(file_path.name)
    return sorted(csv_files)

def find_compatible_datasets(data_dir="/data/gold", debug=False) -> Dict[str, DatasetInfo]:
    """
    Find all processed datasets that contain the 'meta_er_status' column and discover their available CSV files.
    
    Returns a dictionary mapping dataset names to DatasetInfo objects.
    """
    data_path = Path(data_dir)
    compatible_datasets = {}
    
    if not data_path.exists():
        print_now(f"❌ Data directory {data_dir} does not exist")
        return {}
    
    # Get a sorted list of all potential datasets (directories)
    potential_datasets = sorted([item.name for item in data_path.iterdir() if item.is_dir()])
    
    print_now(f"🔍 Found {len(potential_datasets)} potential datasets. Checking for compatibility ('meta_er_status')...")

    for dataset_name in potential_datasets:
        dataset_path = data_path / dataset_name
        unadjusted_file = dataset_path / "unadjusted.csv"
        
        # Skip if 'unadjusted.csv' does not exist
        if not unadjusted_file.exists():
            if debug:
                print_now(f"  - Skipping {dataset_name}: missing unadjusted.csv")
            continue

        try:
            # Read only the header to efficiently check for the column
            df_header = pd.read_csv(unadjusted_file, nrows=0)
            if 'meta_er_status' in df_header.columns:
                # Discover all available CSV files
                available_files = discover_csv_files(dataset_path)
                
                # Get shape for reporting
                gene_count = len(df_header.columns) - len([col for col in df_header.columns if col.startswith('meta_')])
                # Use the number of lines in the file for speed
                sample_count = sum(1 for _ in open(unadjusted_file, 'r')) - 1
                
                dataset_info = DatasetInfo(
                    name=dataset_name,
                    path=dataset_path,
                    available_files=available_files,
                    has_meta_er_status=True,
                    sample_count=sample_count,
                    gene_count=gene_count
                )
                
                compatible_datasets[dataset_name] = dataset_info
                print_now(f"  - ✅ Added {dataset_name} ({len(available_files)} CSV files). ")
                print_now(f"  - Unadjusted has {gene_count} genes and {sample_count} samples. ")
                
                if debug:
                    print_now(f"    Available files: {', '.join(available_files)}")
            else:
                if debug:
                    print_now(f"  - ❌ Skipping {dataset_name}: missing 'meta_er_status' column")
                    print_now(f"    Available metadata columns: {', '.join([col for col in df_header.columns if col.startswith('meta_')])}")
        except Exception as e:
            if debug:
                print_now(f"  - ❌ Skipping {dataset_name}: error reading file -> {e}")

    return compatible_datasets



def validate_file_compatibility(dataset1_info: DatasetInfo, dataset2_info: DatasetInfo, csv_file: str) -> bool:
    """
    Validate that both datasets have the specified CSV file and are compatible for combination.
    """
    return csv_file in dataset1_info.available_files and csv_file in dataset2_info.available_files



def run_combination(dataset1: str, dataset2: str, csv_file: str, output_dir: str, 
                   debug: bool = False, dry_run: bool = False) -> CombinationResult:
    """Run the combine_datasets.py script for a specific CSV file type."""
    
    # Define paths
    input1 = f"/data/gold/{dataset1}/{csv_file}"
    input2 = f"/data/gold/{dataset2}/{csv_file}"
    
    # Create combination name (alphabetical order for consistency)
    combo_name = f"{min(dataset1, dataset2)}_{max(dataset1, dataset2)}"
    output_file = Path(f"{output_dir}/{combo_name}/{csv_file}")
    
    # Create result object
    result = CombinationResult(
        combo_name=combo_name,
        dataset1=dataset1,
        dataset2=dataset2,
        file_type=csv_file,
        success=False,
        output_path=output_file,
    )
    
    if dry_run:
        result.success = True
        return result
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "python3", "/scripts/prepdata/combine_datasets.py",
        "--input1", input1,
        "--input2", input2,
        "--output", str(output_file)
    ]
    
    try:
        # Run the combination script
        process_result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if process_result.returncode == 0:
            # Check if output file was created and has reasonable size
            if output_file.exists() and output_file.stat().st_size > 1000:
                result.file_size = output_file.stat().st_size
                result.success = True
                # Store the output from combine_datasets.py for later display
                if process_result.stdout.strip():
                    result.error_message = process_result.stdout.strip()  # Reuse field for success message
            else:
                result.error_message = "Output file missing or too small"
        else:
            result.error_message = f"Command failed (exit code {process_result.returncode})"
            # Include the error output from combine_datasets.py
            if process_result.stdout.strip():
                result.error_message += f": {process_result.stdout.strip()}"
                
    except subprocess.TimeoutExpired:
        result.error_message = "Timeout (>5 minutes)"
    except Exception as e:
        result.error_message = f"Error: {e}"
    
    return result

def process_single_combination(combination_args):
    """Process a single combination - designed for parallel execution."""
    dataset1, dataset2, csv_file, output_dir, debug  = combination_args
    
    result = run_combination(
        dataset1, dataset2, csv_file, output_dir,
        debug=debug, dry_run=False
    )
    

    
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Generate all possible pairwise combinations of compatible (ER+) datasets for multiple file types.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug output')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually doing it')
    parser.add_argument('--output-dir', default='/data/paired_datasets', help='Output directory for combined datasets')
    parser.add_argument('--data-dir', default='/data/gold', help='Input directory containing processed datasets')
    parser.add_argument('--max-combinations', type=int, help='Maximum number of combinations to process (for testing)')
    parser.add_argument('--csv-files', nargs='*', help='Specific CSV files to process (default: all available)')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel processes (default: 1)')

    
    args = parser.parse_args()
    
    print_now("="*80)
    print_now("GENERATING ALL DATASET COMBINATIONS")
    print_now("="*80)
    
    # Initialize performance tracking
    perf_stats = PerformanceStats()
    perf_stats.start_time = time.time()
    

    
    if not check_disk_space(Path(args.output_dir)):
        print_now("⚠️  Warning: Low disk space detected")
    
    # Optimize parallel processing based on system resources
    if args.parallel > 1:
        cpu_count = os.cpu_count() or 1
        recommended_parallel = min(args.parallel, cpu_count, 8)  # Cap at 8
        if recommended_parallel != args.parallel:
            print_now(f"🔧 Adjusting parallel processes from {args.parallel} to {recommended_parallel}")
            args.parallel = recommended_parallel

    
    # Find all datasets that are compatible for combination
    datasets = find_compatible_datasets(args.data_dir, debug=args.debug)
    
    if not datasets:
        print_now("\n❌ No compatible datasets found with 'meta_er_status' column!")
        return 1
    
    print_now(f"\n✅ Found {len(datasets)} compatible datasets to be combined:")
    for i, (name, info) in enumerate(datasets.items(), 1):
        print_now(f"  {i:2d}. {name} ({len(info.available_files)} CSV files)")
    
    # Determine which CSV files to process
    if args.csv_files:
        csv_files_to_process = args.csv_files
        print_now(f"\n📋 Processing specified CSV files: {', '.join(csv_files_to_process)}")
    else:
        # Find all CSV files that exist in at least 2 datasets
        all_csv_files = set()
        for dataset_info in datasets.values():
            all_csv_files.update(dataset_info.available_files)
        
        csv_files_to_process = []
        for csv_file in sorted(all_csv_files):
            datasets_with_file = [name for name, info in datasets.items() if csv_file in info.available_files]
            if len(datasets_with_file) >= 2:
                csv_files_to_process.append(csv_file)
        
        print_now(f"\n📋 Found {len(csv_files_to_process)} CSV file types that can be combined:")
        for csv_file in csv_files_to_process:
            datasets_with_file = [name for name, info in datasets.items() if csv_file in info.available_files]
            print_now(f"  - {csv_file} (available in {len(datasets_with_file)} datasets)")
    
    # Generate all possible dataset combinations
    dataset_names = list(datasets.keys())
    all_dataset_combinations = list(combinations(dataset_names, 2))
    
    if args.max_combinations and args.max_combinations < len(all_dataset_combinations):
        all_dataset_combinations = all_dataset_combinations[:args.max_combinations]
        print_now(f"🔧 Limited to first {args.max_combinations} dataset combinations for testing")
    
    # Calculate total combinations (dataset pairs × CSV files)
    total_combinations = 0
    valid_combinations = []
    
    for dataset1, dataset2 in all_dataset_combinations:
        for csv_file in csv_files_to_process:
            if validate_file_compatibility(datasets[dataset1], datasets[dataset2], csv_file):
                valid_combinations.append((dataset1, dataset2, csv_file))
                total_combinations += 1
    
    print_now(f"\n📊 Total valid combinations: {total_combinations}")
    print_now(f"    Dataset pairs: {len(all_dataset_combinations)}")
    print_now(f"    CSV file types: {len(csv_files_to_process)}")
    
    if args.dry_run:
        print_now("\n🔍 DRY RUN - Showing combinations that would be created:")
        for i, (dataset1, dataset2, csv_file) in enumerate(valid_combinations[:20], 1):  # Show first 20
            combo_name = f"{min(dataset1, dataset2)}_{max(dataset1, dataset2)}"
            print_now(f"  {i:3d}. {combo_name}/{csv_file}")
        if len(valid_combinations) > 20:
            print_now(f"  ... and {len(valid_combinations) - 20} more")
        return 0
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print_now(f"📁 Output directory: {args.output_dir}")
    
    # Process combinations
    print_now(f"\n🚀 Processing {total_combinations} combinations...")
    if args.parallel > 1:
        print_now(f"⚡ Using {args.parallel} parallel processes")
    
    successful_results = []
    failed_results = []
    
    if args.parallel > 1 and CONCURRENT_FUTURES_AVAILABLE:
        # True parallel processing - collect results in batches to avoid interleaving
        print_now(f"⚡ Processing {args.parallel} combinations simultaneously")
        
        from concurrent.futures import ProcessPoolExecutor
        
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all combinations as individual tasks
            future_to_combo = {}
            for i, (dataset1, dataset2, csv_file) in enumerate(valid_combinations):
                combo_args = (dataset1, dataset2, csv_file, args.output_dir, args.debug)
                future = executor.submit(process_single_combination, combo_args)
                future_to_combo[future] = (i + 1, dataset1, dataset2, csv_file)
            
            # Process results as they complete
            completed = 0
            
            for future in as_completed(future_to_combo):
                combo_idx, dataset1, dataset2, csv_file = future_to_combo[future]
                try:
                    result = future.result()
                    completed += 1
                    
                    if result.success:
                        successful_results.append(result)
                        status = f"✅ {result.combo_name}/{csv_file} ({result.file_size/1024:.1f}KB)"
                        if result.error_message:  # Success message from combine_datasets.py
                            status += f" | {result.error_message}"
                        perf_stats.total_size_mb += result.file_size / (1024 * 1024)
                    else:
                        failed_results.append(result)
                        status = f"❌ {result.combo_name}/{csv_file} | {result.error_message}"
                    
                    print_now(f"[{completed:3d}/{total_combinations}] {status}")
                    
                except Exception as e:
                    completed += 1
                    print_now(f"[{completed:3d}/{total_combinations}] ❌ Exception: {dataset1}+{dataset2}/{csv_file} - {e}")
                    
                    # Create a failed result for tracking
                    failed_result = CombinationResult(
                        combo_name=f"{min(dataset1, dataset2)}_{max(dataset1, dataset2)}",
                        dataset1=dataset1,
                        dataset2=dataset2,
                        file_type=csv_file,
                        success=False,
                        output_path=Path(f"{args.output_dir}/{min(dataset1, dataset2)}_{max(dataset1, dataset2)}/{csv_file}"),
                        error_message=str(e)
                    )
                    failed_results.append(failed_result)
                
                perf_stats.total_files_processed += 1
                
    elif args.parallel > 1:
        print_now("⚠️  Parallel processing requested but concurrent.futures not available, falling back to sequential")
        args.parallel = 1
    
    if args.parallel <= 1:
        # Sequential processing
        for i, (dataset1, dataset2, csv_file) in enumerate(valid_combinations, 1):
            print_now(f"\n[{i:3d}/{total_combinations}]", end="")
            
            result = run_combination(
                dataset1, dataset2, csv_file, args.output_dir, 
                debug=args.debug, dry_run=False
            )
            
            if result.success:
                successful_results.append(result)
                perf_stats.total_size_mb += result.file_size / (1024 * 1024)
            else:
                failed_results.append(result)
            
            perf_stats.total_files_processed += 1
            

    
    # Finalize performance stats
    perf_stats.end_time = time.time()
    perf_stats.total_files_processed = len(successful_results) + len(failed_results)
    
    # Calculate total size for successful results
    if not perf_stats.total_size_mb:  # In case parallel processing didn't update this
        perf_stats.total_size_mb = sum(r.file_size for r in successful_results) / (1024 * 1024)
    

    
    # Summary
    print_now("\n" + "="*80)
    print_now("COMBINATION SUMMARY")
    print_now("="*80)
    print_now(f"✅ Successful combinations: {len(successful_results)}")
    print_now(f"❌ Failed combinations: {len(failed_results)}")
    print_now(f"📊 Success rate: {len(successful_results)/total_combinations*100:.1f}%")
    
    # Performance statistics
    print_now(f"\n⚡ Performance Statistics:")
    print_now(f"   Total processing time: {perf_stats.elapsed_time:.1f}s")

    print_now(f"   Total data processed: {perf_stats.total_size_mb:.1f}MB")
    if perf_stats.elapsed_time > 0:
        print_now(f"   Throughput: {perf_stats.throughput_mb_per_sec:.1f}MB/s")
    
    if successful_results:
        print_now(f"\n📁 Combined datasets saved to: {args.output_dir}")
        
        # Group results by combination name
        combo_groups = {}
        for result in successful_results:
            if result.combo_name not in combo_groups:
                combo_groups[result.combo_name] = []
            combo_groups[result.combo_name].append(result)
        
        print_now(f"✅ Successful combination directories ({len(combo_groups)}):")
        for combo_name in sorted(combo_groups.keys())[:10]:  # Show first 10
            results = combo_groups[combo_name]
            total_size = sum(r.file_size for r in results) / (1024*1024)  # MB
            csv_files = [r.file_type for r in results]
            print_now(f"  - {combo_name}/ ({len(csv_files)} files, {total_size:.1f}MB)")
        
        if len(combo_groups) > 10:
            print_now(f"  ... and {len(combo_groups) - 10} more combination directories")
    
    if failed_results:
        print_now(f"\n❌ Failed combinations:")
        # Group failures by error type
        error_groups = {}
        for result in failed_results:
            error = result.error_message or "Unknown error"
            if error not in error_groups:
                error_groups[error] = []
            error_groups[error].append(result)
        
        for error, results in error_groups.items():
            print_now(f"  {error}: {len(results)} combinations")
            for result in results[:3]:  # Show first 3 examples
                print_now(f"    - {result.combo_name}/{result.file_type}")
            if len(results) > 3:
                print_now(f"    ... and {len(results) - 3} more")
    
    return 0 if not failed_results else 1

if __name__ == "__main__":
    sys.exit(main())