#!/usr/bin/env python3
"""
Modular Dataset Download Script for Breast Cancer Gene Expression Data Integration Project

This script downloads datasets from OSF and Google Drive with configurable source IDs,
organizing them through a clean pipeline: raw_download → raw_data → data

Usage:
    # Download from OSF
    pyth.py --source osf --project-id eky3p --datasets GSE20194,GSE24080
    
    # Download from Google Drive  
    python download_datasets.py --source gdrive --folder-id 1smhpktMRyP4yyFHKHSisxRd9jwb8kvrq --datasets GSE115577,GSE123845
    
    # Download and organize in one step
    python download_datasets.py --source osf --project-id eky3p --organize
    
    # Only organize existing downloads
    python download_datasets.py --organize-only
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from config import DownloadConfig
from downloaders import OSFDownloader, GDriveDownloader
from analyzers import ContentAnalyzer
from utils import print_now


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and organize genomics datasets from OSF or Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Source configuration
    parser.add_argument('--source', choices=['osf', 'gdrive'], 
                       help='Data source type (osf or gdrive)')
    parser.add_argument('--project-id', 
                       help='OSF project ID (required for OSF source)')
    parser.add_argument('--folder-id', 
                       help='Google Drive folder ID (required for gdrive source)')
    
    # Dataset selection
    parser.add_argument('--datasets', 
                       help='Comma-separated list of dataset IDs to download (e.g., GSE20194,GSE24080)')
    
    # Directory configuration
    parser.add_argument('--raw-download-dir', type=Path, help='Directory for raw downloads')
    parser.add_argument('--raw-data-dir', type=Path, help='Directory for organized data')
    
    # Configuration options
    parser.add_argument('--max-retries', type=int, default=2,
                       help='Maximum retry attempts for failed downloads')
    parser.add_argument('--timeout', type=int, default=20,
                       help='Timeout in seconds for download requests')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command-line arguments."""
    if not args.raw_data_dir:
        print_now("❌ Error: --raw-data-dir is required")
        return False

    if not args.raw_download_dir:
        print_now("❌ Error: --raw-download-dir is required")
        return False
    
    if not args.source:
        print_now("❌ Error: --source is required (osf or gdrive)")
        return False
    
    if args.source == 'osf' and not args.project_id:
        print_now("❌ Error: --project-id is required for OSF source")
        return False
    
    if args.source == 'gdrive' and not args.folder_id:
        print_now("❌ Error: --folder-id is required for Google Drive source")
        return False
    
    return True


def create_config(args) -> DownloadConfig:
    """Create download configuration from arguments."""
    # Parse datasets list
    datasets = None
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
    
    # Determine source ID
    source_id = args.project_id if args.source == 'osf' else args.folder_id
    
    return DownloadConfig(
        source_type=args.source,
        source_id=source_id,
        datasets=datasets,
        raw_download_dir=args.raw_download_dir,
        raw_data_dir=args.raw_data_dir,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout
    )


def download_phase(config: DownloadConfig) -> bool:
    """Download files from specified source to raw_download directory."""
    print_now(f"🔄 Starting download phase from {config.source_type.upper()}")
    
    try:
        # Create appropriate downloader
        if config.source_type == 'osf':
            downloader = OSFDownloader(config)
        elif config.source_type == 'gdrive':
            downloader = GDriveDownloader(config)
        else:
            print_now(f"❌ Unsupported source type: {config.source_type}")
            return False
        
        # Determine datasets to download
        datasets_to_download = config.datasets
        
        if not datasets_to_download:
            print_now("⚠️  No datasets specified for download")
            return False
        
        print_now(f"🎯 Downloading {len(datasets_to_download)} datasets: {datasets_to_download}")
        
        # Download each dataset
        success_count = 0
        with downloader:
            for dataset_id in datasets_to_download:
                if downloader.download_dataset(dataset_id):
                    success_count += 1
                    print_now(f"✅ Successfully downloaded dataset: {dataset_id}")
                else:
                    print_now(f"❌ Failed to download dataset: {dataset_id}")
        
        print_now(f"📊 Download phase complete: {success_count}/{len(datasets_to_download)} datasets downloaded")
        return success_count > 0
        
    except Exception as e:
        print_now(f"❌ Error in download phase: {e}")
        return False


def main():
    """Main entry point with flexible argument parsing."""
    args = parse_arguments()
    
    if not validate_arguments(args):
        sys.exit(1)
    
    config = create_config(args)
    
    print_now("🚀 Starting modular dataset downloader")
    print_now(f"   Raw download directory: {config.raw_download_dir}")
    print_now(f"   Raw data directory: {config.raw_data_dir}")
    
    if download_phase(config):
        print_now("✅ Download phase completed successfully")
    else:
        print_now("❌ Download phase failed")
        sys.exit(1)


if __name__ == "__main__":
    main()