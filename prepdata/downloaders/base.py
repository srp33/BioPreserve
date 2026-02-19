"""
Base downloader interface with retry logic and error handling.
"""

import time
import requests
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
import gzip
import shutil

try:
    from ..config import DownloadConfig
    from ..utils import NetworkError, FileSystemError, print_now
except ImportError:
    from config import DownloadConfig
    from utils import NetworkError, FileSystemError, print_now


class BaseDownloader(ABC):
    """Abstract base class for dataset downloaders."""
    
    def __init__(self, config: DownloadConfig):
        """Initialize with download configuration."""
        self.config = config
        self.session = requests.Session()
        
        # Create directories
        self.config.raw_download_dir.mkdir(parents=True, exist_ok=True)
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup session."""
        self.session.close()
    
    @abstractmethod
    def list_available_files(self) -> Dict[str, Dict]:
        """Get all available files from the source."""
        pass
    
    @abstractmethod
    def find_dataset_files(self, dataset_id: str) -> List[Dict]:
        """Find files matching dataset ID pattern."""
        pass
    
    def download_dataset(self, dataset_id: str) -> bool:
        """Download all files for a dataset."""
        try:
            print_now(f"📥 Downloading dataset: {dataset_id}")
            
            # Find files for this dataset
            files = self.find_dataset_files(dataset_id)
            if not files:
                print_now(f"⚠️  No files found for dataset {dataset_id}")
                return False
            
            # Create dataset directory
            dataset_dir = self.config.raw_download_dir / dataset_id.lower()
            print_now(f"Dataset dir: {dataset_dir}")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            success_count = 0
            for file_info in files:
                file_name = file_info.get('unique_name', 'unknown_file')
                download_url = file_info.get('download_url')
                
                if not download_url:
                    print_now(f"⚠️  No download URL for {file_name}")
                    continue
                
                target_path = dataset_dir / file_name
                print_now(f"Target_path: {target_path}")
                if self.download_file(download_url, target_path):
                    success_count += 1
                else:
                    print_now(f"❌ Failed to download {file_name}")
            
            print_now(f"✅ Downloaded {success_count}/{len(files)} files for {dataset_id}")
            return success_count > 0
            
        except Exception as e:
            print_now(f"❌ Error downloading dataset {dataset_id}: {e}")
            return False
    
    def download_file(self, url: str, target_path: Path) -> bool:
        """Download single file with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                if attempt > 0:
                    delay = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                    print_now(f"⏱️  Waiting {delay}s before retry...")
                    time.sleep(delay)
                
                response = self.session.get(url, stream=True, timeout=self.config.timeout_seconds)
                response.raise_for_status()
                
                # Get file size for verification
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0

                # Integrate url into name (keep suffixes) to assist with uniqueness
                temp_file = self.config.raw_download_dir / f"{url.split('/')[-1]}_{target_path.name}"
                temp_file = temp_file.with_suffix(target_path.suffix + '.tmp')
                
                # Download to temporary file first
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                
                # Verify download completed
                if total_size > 0 and downloaded_size != total_size:
                    print_now(f"⚠️  Download size mismatch: expected {total_size}, got {downloaded_size}")
                    temp_file.unlink(missing_ok=True)
                    continue
                
                # Handle decompression if needed
                if self._is_gzip_file(temp_file):
                    if self._decompress_gzip_file(temp_file, target_path):
                        temp_file.unlink(missing_ok=True)
                        # Remove the .gz suffix, but keep other suffixes
                        unzipped_path = target_path.with_suffix('')
                        target_path.rename(unzipped_path)
                        return True
                    else:
                        temp_file.unlink(missing_ok=True)
                        continue
                else:
                    # Move temp file to final location
                    temp_file.rename(target_path)
                    return True
                    
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise NetworkError(f"Failed to download {url} after {self.config.max_retries} attempts: {e}")
                print_now(f"⚠️  Download attempt {attempt + 1} failed: {e}")
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise FileSystemError(f"File system error downloading {url}: {e}")
                print_now(f"⚠️  File system error on attempt {attempt + 1}: {e}")
        
        return False
    
    def verify_download(self, file_path: Path, expected_size: Optional[int] = None) -> bool:
        """Verify downloaded file integrity."""
        try:
            if not file_path.exists():
                return False
            
            if file_path.stat().st_size == 0:
                return False
            
            if expected_size and file_path.stat().st_size != expected_size:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_gzip_file(self, file_path: Path) -> bool:
        """Check if file is gzip compressed by reading magic bytes."""
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(2)
                return magic == b'\x1f\x8b'
        except Exception:
            return False
    
    def _decompress_gzip_file(self, gzip_path: Path, output_path: Path) -> bool:
        """Decompress gzip file to output path."""
        try:
            with gzip.open(gzip_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True
        except Exception as e:
            print_now(f"❌ Failed to decompress {gzip_path}: {e}")
            return False