"""
Content analyzer for determining file types and extracting dataset information.
"""

import re
import gzip
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from io import StringIO
import pandas as pd

try:
    from ..config import FileType
    from ..utils import AnalysisError
except ImportError:
    from config import FileType
    from utils import AnalysisError


@dataclass
class AnalysisResult:
    """Result of file content analysis."""
    file_path: Path
    content_type: FileType
    detected_datasets: List[str]
    confidence_score: float
    file_size_mb: float
    # Fields for tabular data, optional for non-tabular files
    sample_count: Optional[int] = None
    feature_count: Optional[int] = None
    has_numeric_data: Optional[bool] = None
    has_sample_ids: Optional[bool] = None
    error_message: Optional[str] = None


class ContentAnalyzer:
    """Analyzer for determining file content type and characteristics."""

    def __init__(self, debug: bool = False):
        """Initialize content analyzer."""
        self.debug = debug

        # Regex patterns for dataset ID detection
        self.gse_patterns = [
            r'GSE\d+[A-Z]*',
            r'gse\d+[a-z]*',
            r'GSE_\d+',
            r'gse_\d+',
        ]

        # Expression data indicators
        self.expression_indicators = [
            'entrez_gene_id', 'hgnc_symbol', 'ensembl_gene_id', 'gene_symbol',
            'probe_id', 'probeset_id', 'gene_id', 'transcript_id',
            'chromosome', 'gene_biotype', 'gene_name'
        ]

        # Metadata indicators
        self.metadata_indicators = [
            'sample_id', 'patient_id', 'age', 'grade', 'stage', 'survival',
            'platform_id', 'series', 'treatment', 'outcome', 'clinical',
            'er', 'pr', 'her2', 'node', 'size', 'dmfs', 'rfs'
        ]

        # Sample ID patterns (common in genomics)
        self.sample_id_patterns = [
            r'GSM\d+',
            r'TCGA-[A-Z0-9-]+',
            r'[A-Z]+\d+[A-Z]*',
        ]


    def _fraction_short_rows(self, df) -> float:
        if df is None or df.empty:
            return 1.0
        col_counts = df.notna().sum(axis=1)
        return (col_counts < col_counts.max()).sum() / len(col_counts)


    def analyze_file(self, file_path: Path, sample_lines: int = 100) -> AnalysisResult:
        """
        Analyze a file to determine its content type and extract dataset information.
        Simplified Logic:
        1. Attempt to parse as multi-column tabular data.
        2. If successful, classify as EXPRESSION or METADATA.
        3. If parsing fails or results in a single column, classify as NON_TABULAR.
        """
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            content_sample = self._read_file_sample(file_path, sample_lines)

            if not content_sample:
                return self._create_error_result(file_path, file_size_mb, "Could not read file content")

            # --- Simplified Logic: Attempt Tabular Analysis ---
            try:
                df = self._parse_sample_as_string(content_sample)
                short_rows_fraction = self._fraction_short_rows(df)

                # A file is considered tabular only if it has more than one column, and if it has a low number of malformed rows.
                if df is not None and not df.empty and df.shape[1] > 1 and short_rows_fraction < 0.25:
                    df = self._parse_content_sample(content_sample)
                    analysis = self._analyze_dataframe(df, file_path, file_size_mb)
                    detected_datasets = self._extract_dataset_ids_from_dataframe(df)
                    analysis.detected_datasets = detected_datasets
                    return analysis
                
                # If it parsed but has only one column, or is empty, treat as non-tabular.
                if self.debug and df is not None:
                    print(f"DEBUG: Parsed {file_path.name}, but it is not multi-column tabular (shape: {df.shape}). Short row fraction: {short_rows_fraction}. Classifying as NON_TABULAR.")

            except AnalysisError as e:
                # This is the expected path for files that are not CSV/TSV.
                if self.debug:
                    print(f"DEBUG: Failed to parse {file_path.name} as tabular. Error: {e}. Classifying as NON_TABULAR.")

            # --- Fallback for all non-tabular cases ---
            return AnalysisResult(
                file_path=file_path,
                content_type=FileType.NON_TABULAR,
                detected_datasets=self.extract_dataset_ids(file_path),
                confidence_score=1.0,  # We are confident it's not multi-column tabular.
                file_size_mb=file_size_mb,
                error_message="File could not be parsed as multi-column tabular data."
            )

        except Exception as e:
            if self.debug:
                print(f"DEBUG: An unexpected error occurred in analyze_file: {e}")
            return self._create_error_result(file_path, 0.0, str(e))

    def _create_error_result(self, file_path, file_size_mb, error_message):
        """Helper to create a default AnalysisResult for errors."""
        return AnalysisResult(
            file_path=file_path,
            content_type=FileType.UNKNOWN,
            detected_datasets=[],
            confidence_score=0.0,
            file_size_mb=file_size_mb,
            error_message=error_message
        )

    def extract_dataset_ids(self, file_path: Path) -> List[str]:
        """Extract GSE IDs from filename and content."""
        detected_ids = set()
        filename = file_path.name
        for pattern in self.gse_patterns:
            detected_ids.update(re.findall(pattern, filename, re.IGNORECASE))

        try:
            content_sample = self._read_file_sample(file_path, 20)
            for pattern in self.gse_patterns:
                detected_ids.update(re.findall(pattern, content_sample, re.IGNORECASE))
        except Exception:
            pass  # Ignore content extraction errors

        return sorted([pid.upper() for pid in detected_ids])

    def _read_file_sample(self, file_path: Path, sample_lines: int) -> Tuple[str, str]:
        """Read a sample of lines from the file."""
        content_lines = []
        try:
            open_func = gzip.open if file_path.suffix.lower() == '.gz' else open
            mode = 'rt'
            with open_func(file_path, mode, encoding='utf-8', errors='ignore') as f:
                for _ in range(sample_lines):
                    line = f.readline()
                    if not line:
                        break
                    content_lines.append(line)
            content = ''.join(content_lines)
            return content
        except Exception as e:
            if self.debug:
                print(f"DEBUG: ⚠️  Error reading file sample: {e}")
            return ""

    def _parse_sample_as_string(self, content: str) -> Optional[pd.DataFrame]:
        """Parse content sample into a DataFrame, raising an error on failure."""
        if not content.strip():
            return None
        try:
            return pd.read_csv(StringIO(content), sep=None, header=None, na_filter=False, engine='python', dtype='str')
        except Exception as e:
            raise AnalysisError(f"Could not parse content as tabular data") from e

    def _parse_content_sample(self, content: str) -> Optional[pd.DataFrame]:
        """Parse content sample into a DataFrame, raising an error on failure."""
        if not content.strip():
            return None
        try:
            return pd.read_csv(StringIO(content), sep=None, engine='python')
        except Exception as e:
            raise AnalysisError(f"Could not parse content as tabular data") from e

    def _analyze_dataframe(self, df: pd.DataFrame, file_path: Path, file_size_mb: float) -> AnalysisResult:
        """Analyze a DataFrame to determine content type and characteristics."""
        content_type, confidence = self._classify_tabular_content_type(df, file_path.name)

        return AnalysisResult(
            file_path=file_path,
            content_type=content_type,
            detected_datasets=[],
            confidence_score=confidence,
            sample_count=len(df),
            feature_count=len(df.columns),
            has_numeric_data=any(df.select_dtypes(include=['number'])),
            has_sample_ids=self._detect_sample_ids(df),
            file_size_mb=file_size_mb
        )

    def _classify_tabular_content_type(self, df: pd.DataFrame, filename: str) -> Tuple[FileType, float]:
        """Classify the content type based on column names, data patterns, and filename."""
        columns_lower = [str(col).lower() for col in df.columns]
        expression_score = sum(any(indicator in col for col in columns_lower) for indicator in self.expression_indicators)
        metadata_score = sum(any(indicator in col for col in columns_lower) for indicator in self.metadata_indicators)

        numeric_ratio = len(df.select_dtypes(include=['number']).columns) / len(df.columns)
        if numeric_ratio > 0.98:
            expression_score += 4
        elif numeric_ratio < 0.85:
            metadata_score += 4

        first_col_sample = df.iloc[:10, 0].astype(str).str.lower()
        if first_col_sample.str.contains('gene|ensg|probe').any():
            expression_score += 2

        filename_lower = filename.lower()
        if 'expression' in filename_lower or 'expr' in filename_lower:
            expression_score += 3
        elif 'meta' in filename_lower or 'clinical' in filename_lower:
            metadata_score += 3

        print(f"Expression score: {expression_score}, Metadata score: {metadata_score}, Numeric ratio: {numeric_ratio:.4f}")

        total_score = expression_score + metadata_score
        if total_score == 0:
            return FileType.UNKNOWN, 0.1

        if expression_score > metadata_score:
            confidence = expression_score / total_score
            return FileType.EXPRESSION, min(confidence, 0.95)
        else:
            confidence = metadata_score / total_score
            return FileType.METADATA, min(confidence, 0.95)

    def _detect_sample_ids(self, df: pd.DataFrame) -> bool:
        """Detect if the DataFrame contains sample ID patterns."""
        for col in df.columns:
            if any(re.search(pattern, str(col)) for pattern in self.sample_id_patterns):
                return True

        if len(df.columns) > 0 and len(df) > 0:
            first_col_sample = df.iloc[:20, 0].dropna().astype(str)
            if any(any(re.search(pattern, val) for pattern in self.sample_id_patterns) for val in first_col_sample):
                return True

        return False

    def _extract_dataset_ids_from_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Extract dataset IDs from DataFrame content."""
        detected_ids = set()
        for col in df.select_dtypes(include=['object']).columns:
            sample_values = df[col].dropna().astype(str).head(20)
            for val in sample_values:
                for pattern in self.gse_patterns:
                    detected_ids.update(re.findall(pattern, val, re.IGNORECASE))

        for col in df.columns:
            for pattern in self.gse_patterns:
                detected_ids.update(re.findall(pattern, str(col), re.IGNORECASE))

        return sorted([pid.upper() for pid in detected_ids])
