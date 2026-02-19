"""
Utility functions for error handling and logging.
"""

import functools
import time
from pathlib import Path
from typing import Callable, Any

def print_now(*args, **kwargs):
    print(*args, **kwargs, flush=True)

class DownloadError(Exception):
    """Base class for download-related errors."""
    pass

class NetworkError(DownloadError):
    """Network-related download failures."""
    pass

class FileSystemError(DownloadError):
    """File system operation failures."""
    pass

class AnalysisError(Exception):
    """Content analysis failures."""
    pass

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0, max_delay: float = 10.0):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries - 1:
                        # Last attempt, re-raise the exception
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(backoff_factor ** attempt, max_delay)
                    print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                    print(f"⏱️  Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def safe_file_operation(operation_name: str):
    """Decorator for safe file operations with cleanup."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            temp_files = []
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Clean up any temporary files created during the operation
                for temp_file in temp_files:
                    try:
                        if isinstance(temp_file, Path) and temp_file.exists():
                            temp_file.unlink()
                    except Exception:
                        pass  # Ignore cleanup errors
                
                raise FileSystemError(f"{operation_name} failed: {e}")
        
        return wrapper
    return decorator


def check_disk_space(path: Path, required_bytes: int) -> bool:
    """Check if there's enough disk space available."""
    try:
        import shutil
        free_bytes = shutil.disk_usage(path).free
        return free_bytes >= required_bytes
    except Exception:
        # If we can't check, assume there's enough space
        return True


def validate_file_integrity(file_path: Path, expected_size: int = None, check_content: bool = True) -> bool:
    """Validate file integrity."""
    try:
        if not file_path.exists():
            return False
        
        # Check file size
        actual_size = file_path.stat().st_size
        if actual_size == 0:
            return False
        
        if expected_size and actual_size != expected_size:
            return False
        
        # Basic content check
        if check_content:
            try:
                with open(file_path, 'rb') as f:
                    # Try to read first few bytes
                    f.read(1024)
                return True
            except Exception:
                return False
        
        return True
        
    except Exception:
        return False


def create_error_report(operation: str, errors: list, successes: list = None) -> str:
    """Create a formatted error report."""
    report_lines = [f"📊 {operation} Summary"]
    
    if successes:
        report_lines.append(f"✅ Successful: {len(successes)}")
        if len(successes) <= 10:  # Show details for small lists
            for item in successes:
                report_lines.append(f"   - {item}")
    
    if errors:
        report_lines.append(f"❌ Failed: {len(errors)}")
        for error in errors[:10]:  # Show first 10 errors
            report_lines.append(f"   - {error}")
        
        if len(errors) > 10:
            report_lines.append(f"   ... and {len(errors) - 10} more errors")
    
    return "\n".join(report_lines)


def log_operation(operation: str, verbose: bool = False):
    """Decorator for logging operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if verbose:
                print(f"🔄 Starting {operation}...")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if verbose:
                    print(f"✅ {operation} completed in {duration:.2f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ {operation} failed after {duration:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator


class ProgressTracker:
    """Simple progress tracker for operations."""
    
    def __init__(self, operation_name: str, total_items: int):
        self.operation_name = operation_name
        self.total_items = total_items
        self.completed_items = 0
        self.start_time = time.time()
        self.errors = []
        self.successes = []
    
    def update(self, success: bool, item_name: str = None, error_msg: str = None):
        """Update progress with success/failure."""
        self.completed_items += 1
        
        if success:
            self.successes.append(item_name or f"Item {self.completed_items}")
        else:
            error_info = f"{item_name}: {error_msg}" if item_name and error_msg else (item_name or error_msg or f"Item {self.completed_items}")
            self.errors.append(error_info)
        
        # Print progress
        percentage = (self.completed_items / self.total_items) * 100
        print(f"📊 {self.operation_name}: {self.completed_items}/{self.total_items} ({percentage:.1f}%)")
    
    def finish(self):
        """Print final summary."""
        duration = time.time() - self.start_time
        success_count = len(self.successes)
        error_count = len(self.errors)
        
        print(f"🏁 {self.operation_name} completed in {duration:.2f}s")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Failed: {error_count}")
        
        if self.errors:
            print("   Recent errors:")
            for error in self.errors[-3:]:  # Show last 3 errors
                print(f"     - {error}")
        
        return success_count, error_count