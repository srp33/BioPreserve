"""
Google Drive downloader implementation.
"""

import re
import tempfile
import shutil
from typing import Dict, List
from pathlib import Path

try:
    from .base import BaseDownloader
    from ..utils import NetworkError, FileSystemError, print_now
except ImportError:
    from downloaders.base import BaseDownloader
    from utils import NetworkError, FileSystemError, print_now


class GDriveDownloader(BaseDownloader):
    """Downloader for Google Drive datasets."""
    
    def __init__(self, config):
        """Initialize with configurable Google Drive folder ID."""
        super().__init__(config)
        self.folder_id = config.source_id
        
        # Check if gdown is available
        try:
            import gdown
            self.gdown = gdown
        except ImportError:
            raise ImportError("gdown library is required for Google Drive downloads. Install with: pip install gdown")
        
        # Dataset ID patterns for matching files
        self.gse_patterns = [
            r'GSE\d+[A-Z]*',  # GSE12345, GSE12345H, GSE12345N
            r'gse\d+[a-z]*',  # gse12345, gse12345h, gse12345n
            r'GSE_\d+',       # GSE_12345
            r'gse_\d+',       # gse_12345
        ]
    
    def list_available_files(self) -> Dict[str, Dict]:
        """Get all available files from Google Drive folder."""
        try:
            print_now(f"🌐 Fetching files from Google Drive folder: {self.folder_id}")
            
            # Create temporary directory for listing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download folder structure to get file list
                folder_url = f"https://drive.google.com/drive/folders/{self.folder_id}"
                
                try:
                    # Use gdown to list folder contents
                    self.gdown.download_folder(folder_url, output=str(temp_path), quiet=True, use_cookies=False)
                    
                    # Scan downloaded files to build file map
                    file_map = {}
                    for file_path in temp_path.rglob('*'):
                        if file_path.is_file():
                            relative_path = file_path.relative_to(temp_path)
                            file_info = {
                                'name': file_path.name,
                                'unique_name':  str(relative_path).replace('/', '_'),
                                'path': str(relative_path),
                                'size': file_path.stat().st_size,
                                'download_url': folder_url,  # We'll handle this specially
                                'local_path': file_path  # Temporary local path
                            }
                            file_map[file_path.name] = file_info
                    
                    print_now(f"📊 Found {len(file_map)} files in Google Drive folder")
                    return file_map
                    
                except Exception as e:
                    print_now(f"⚠️  Failed to list Google Drive folder contents: {e}")
                    return {}
                    
        except Exception as e:
            raise NetworkError(f"Failed to access Google Drive folder: {e}")
    
    def find_dataset_files(self, dataset_id: str) -> List[Dict]:
        """Find files matching dataset ID pattern."""
        all_files = self.list_available_files()
        matching_files = []
        
        # Normalize dataset ID for comparison
        dataset_id_lower = dataset_id.lower()
        
        for filename, file_info in all_files.items():
            filename_lower = filename.lower()
            
            # Direct filename match
            if dataset_id_lower in filename_lower:
                matching_files.append(file_info)
                continue
            
            # Pattern-based matching
            for pattern in self.gse_patterns:
                matches = re.findall(pattern, filename, re.IGNORECASE)
                for match in matches:
                    if match.upper() == dataset_id.upper():
                        matching_files.append(file_info)
                        break
        
        print_now(f"🎯 Found {len(matching_files)} files for dataset {dataset_id}")
        return matching_files
    
    def download_file(self, url: str, target_path: Path) -> bool:
        """Download single file from Google Drive."""
        # For Google Drive, we handle this differently since we already have local files
        # from the folder download. This method is overridden to handle the special case.
        try:
            # Create parent directory
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # If this is a direct file URL, use gdown
            if 'drive.google.com' in url and 'file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
                return self._download_file_by_id(file_id, target_path)
            else:
                # This is likely a folder download case - handle in download_dataset
                return super().download_file(url, target_path)
                
        except Exception as e:
            print_now(f"❌ Error downloading from Google Drive: {e}")
            return False
    
    def download_dataset(self, dataset_id: str) -> bool:
        """Download all files for a dataset from Google Drive."""
        try:
            print_now(f"📥 Downloading dataset from Google Drive: {dataset_id}")
            
            # Create dataset directory
            dataset_dir = self.config.raw_download_dir / dataset_id.lower()
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Download entire folder to temporary location
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                folder_url = f"https://drive.google.com/drive/folders/{self.folder_id}"
                
                try:
                    self.gdown.download_folder(folder_url, output=str(temp_path), quiet=False, use_cookies=False)
                    
                    # Find and copy files matching the dataset
                    matching_files = []
                    dataset_id_lower = dataset_id.lower()
                    
                    for file_path in temp_path.rglob('*'):
                        if file_path.is_file():
                            filename_lower = file_path.name.lower()
                            
                            # Check if file matches dataset
                            if dataset_id_lower in filename_lower:
                                matching_files.append(file_path)
                                continue
                            
                            # Pattern-based matching
                            for pattern in self.gse_patterns:
                                matches = re.findall(pattern, file_path.name, re.IGNORECASE)
                                for match in matches:
                                    if match.upper() == dataset_id.upper():
                                        matching_files.append(file_path)
                                        break
                    
                    # Copy matching files to dataset directory
                    success_count = 0
                    for file_path in matching_files:
                        target_path = dataset_dir / file_path.name
                        try:
                            shutil.copy2(file_path, target_path)
                            success_count += 1
                            print_now(f"✅ Copied {file_path.name}")
                        except Exception as e:
                            print_now(f"❌ Failed to copy {file_path.name}: {e}")
                    
                    print_now(f"✅ Downloaded {success_count}/{len(matching_files)} files for {dataset_id}")
                    return success_count > 0
                    
                except Exception as e:
                    print_now(f"❌ Failed to download Google Drive folder: {e}")
                    return False
                    
        except Exception as e:
            print_now(f"❌ Error downloading dataset {dataset_id}: {e}")
            return False
    
    def _download_file_by_id(self, file_id: str, target_path: Path) -> bool:
        """Download a specific file by Google Drive file ID."""
        try:
            file_url = f"https://drive.google.com/uc?id={file_id}"
            self.gdown.download(file_url, str(target_path), quiet=False)
            return target_path.exists() and target_path.stat().st_size > 0
        except Exception as e:
            print_now(f"❌ Failed to download file {file_id}: {e}")
            return False