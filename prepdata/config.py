#!/usr/bin/env python3
"""
Configuration module for the modular dataset downloader.

This module defines configuration classes, enums, and constants used throughout
the dataset downloading and organization system.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any


# File type classification enum
class FileType(Enum):
    """Enumeration for different file types in dataset analysis."""
    UNKNOWN = 'unknown'
    EXPRESSION = 'expression'
    METADATA = 'metadata'
    NON_TABULAR = 'non_tabular'



@dataclass
class DownloadConfig:
    """Configuration for dataset downloading operations."""
    source_type: str  # 'osf' or 'gdrive'
    source_id: str    # project_id for OSF, folder_id for Google Drive
    datasets: Optional[List[str]] = None  # specific datasets to download, None for all
    raw_download_dir: Path = Path("data/raw_download")
    raw_data_dir: Path = Path("data/raw_data")
    max_retries: int = 3
    timeout_seconds: int = 60
    chunk_size: int = 8192


@dataclass
class DatasetInfo:
    """Information about a dataset to be downloaded."""
    dataset_id: str
    source: str  # 'osf' or 'gdrive'
    source_id: str  # specific project/folder ID
    files: List[str]
    status: str  # 'pending', 'downloading', 'complete', 'failed'


@dataclass
class AnalysisResult:
    """Result of content analysis for a file."""
    file_path: Path
    content_type: str  # 'expression', 'metadata', 'unknown'
    dataset_ids: List[str]
    confidence: float
    sample_count: int
    feature_count: int


@dataclass
class FileInfo:
    """Information about a file being organized."""
    original_path: Path
    dataset_id: str
    file_type: FileType
    confidence: float
    target_name: str  # e.g., "expression_GSE20194.csv"
