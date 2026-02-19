"""
OSF (Open Science Framework) downloader implementation.
"""

import re
import time
from typing import Dict, List
import requests

try:
    from .base import BaseDownloader
    from ..utils import NetworkError, print_now
except ImportError:
    from downloaders.base import BaseDownloader
    from utils import NetworkError, print_now


class OSFDownloader(BaseDownloader):
    """Downloader for OSF (Open Science Framework) datasets."""
    
    def __init__(self, config):
        """Initialize with configurable OSF project ID."""
        super().__init__(config)
        self.project_id = config.source_id
        self.base_url = f"https://api.osf.io/v2/nodes/{self.project_id}/files/osfstorage/"
        
        # Dataset ID patterns for matching files
        self.gse_patterns = [
            r'GSE\d+[A-Z]*',  # GSE12345, GSE12345H, GSE12345N
            r'gse\d+[a-z]*',  # gse12345, gse12345h, gse12345n
            r'GSE_\d+',       # GSE_12345
            r'gse_\d+',       # gse_12345
        ]

        self.available_files = {}  # Cache for available files
    
    def list_available_files(self) -> Dict[str, Dict]:
        """Get all available files from OSF with pagination and recursive folder search."""
        try:
            print_now(f"🌐 Fetching files from OSF project: {self.project_id}")
            all_files_data = self._get_all_osf_data_with_pagination(self.base_url)
            
            file_map = {}
            folders_to_search = []
            
            # First pass: collect files and identify folders
            for file_info in all_files_data:
                name = file_info.get('attributes', {}).get('name', 'Unknown')
                kind = file_info.get('attributes', {}).get('kind', 'file')
                
                # Add download URL from links
                links = file_info.get('links', {})
                download_url = links.get('download')
                if download_url:
                    file_info['download_url'] = download_url
                
                # Set name as top-level field for base downloader compatibility
                file_info['name'] = name
                
                file_map[name] = file_info
                
                # If it's a folder, add to search list
                if kind == 'folder':
                    folders_to_search.append((name, file_info))
            
            # Second pass: recursively search folders
            max_depth = 5
            for folder_name, folder_info in folders_to_search:
                print_now(f"🔍 Searching folder: {folder_name}")
                folder_files = self._search_folder_recursively(folder_info, depth=0, max_depth=max_depth)
                for file_path, file_info in folder_files.items():
                    file_map[f"{folder_name}/{file_path}"] = file_info
            
            print_now(f"📊 Found {len(file_map)} files in OSF project (including subfolders)")
            return file_map
            
        except Exception as e:
            raise NetworkError(f"Failed to list OSF files: {e}")
    
    def find_dataset_files(self, dataset_id: str) -> List[Dict]:
        """Find files matching dataset ID pattern."""
        if not self.available_files:
            self.available_files = self.list_available_files()
            print(self.available_files)
        all_files = self.available_files 

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
    
    def _get_all_osf_data_with_pagination(self, url: str) -> List[dict]:
        """Get all data from OSF API with pagination support."""
        all_data = []
        current_url = url
        page_count = 0
        
        while current_url:
            page_count += 1
            
            # Retry logic for each page
            response = None
            for attempt in range(self.config.max_retries):
                try:
                    response = self.session.get(current_url, timeout=self.config.timeout_seconds)
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == self.config.max_retries - 1:
                        raise NetworkError(f"Failed to fetch OSF page {page_count} after {self.config.max_retries} attempts: {e}")
                    print_now(f"⚠️  Page {page_count} request failed (attempt {attempt + 1}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            if response is None:
                break
            
            try:
                page_data = response.json()
            except ValueError as e:
                print_now(f"❌ Failed to parse JSON response for page {page_count}: {e}")
                break
            
            # Extract data from this page
            page_items = page_data.get('data', [])
            all_data.extend(page_items)
            
            # Get next page URL
            links = page_data.get('links', {})
            next_url = links.get('next')
            if isinstance(next_url, dict):
                next_url = next_url.get('href')
            current_url = next_url
            
            # Safety check to prevent infinite loops
            if page_count > 1000:
                print_now(f"⚠️  Reached maximum page limit (1000), stopping pagination")
                break
            
            # If no items on this page, we might be done
            if len(page_items) == 0:
                break
        
        return all_data
    
    def _search_folder_recursively(self, folder_info: dict, depth: int, max_depth: int) -> Dict[str, dict]:
        """Recursively search a folder for files."""
        if depth >= max_depth:
            return {}
        
        folder_files = {}
        
        try:
            # Get folder contents URL
            relationships = folder_info.get('relationships', {})
            files_rel = relationships.get('files', {})
            files_url = files_rel.get('links', {}).get('related', {}).get('href')
            
            if not files_url:
                return {}
            
            # Get files in this folder
            folder_contents = self._get_all_osf_data_with_pagination(files_url)
            
            for item in folder_contents:
                name = item.get('attributes', {}).get('name', 'Unknown')
                kind = item.get('attributes', {}).get('kind', 'file')
                
                # Add download URL
                links = item.get('links', {})
                download_url = links.get('download')
                if download_url:
                    item['download_url'] = download_url
                
                # Set unique name before recursing for path propagation
                item['unique_name'] = name
                if folder_info.get('unique_name'):
                    item['unique_name'] = f"{folder_info['unique_name']}_{name}"
                
                if kind == 'file':
                    folder_files[name] = item
                elif kind == 'folder' and depth < max_depth - 1:
                    # Recursively search subfolders
                    subfolder_files = self._search_folder_recursively(item, depth + 1, max_depth)
                    for subfile_path, subfile_info in subfolder_files.items():
                        folder_files[f"{name}/{subfile_path}"] = subfile_info
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.1)
            
        except Exception as e:
            print_now(f"⚠️  Error searching folder at depth {depth}: {e}")
        
        return folder_files