"""
Dataset downloaders module for handling different data sources.
"""

try:
    from .base import BaseDownloader
    from .osf_downloader import OSFDownloader
    from .gdrive_downloader import GDriveDownloader
except ImportError:
    # Fallback to absolute imports
    from downloaders.base import BaseDownloader
    from downloaders.osf_downloader import OSFDownloader
    from downloaders.gdrive_downloader import GDriveDownloader

__all__ = ['BaseDownloader', 'OSFDownloader', 'GDriveDownloader']