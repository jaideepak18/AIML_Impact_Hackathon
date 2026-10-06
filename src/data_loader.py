"""Data loading module for Business Entity Resolution.

Provides strict, memory-conscious loading for tab-separated value (.tsv) files
with column validation, missing value imputation, and ground truth parsing.
"""

import logging
import os
from typing import Dict, List, Optional, Set, Union
import pandas as pd

logger = logging.getLogger(__name__)

# Expected schema definitions
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def validate_columns(
    df: pd.DataFrame, expected_columns: List[str], file_name: str
) -> None:
    """Ensure dataframe possesses exact required columns without casing issues."""
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"File '{file_name}' missing expected columns {missing}. "
            f"Found: {list(df.columns)}. Verify sep='\\t' was used."
        )


def load_tsv_file(
    file_path: str,
    expected_columns: List[str],
    dtype_dict: Optional[Dict[str, str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a tab-separated (.tsv) file with column validation and string preservation."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Requested dataset file not found at: {file_path}")

    if dtype_dict is None:
        dtype_dict = {col: str for col in expected_columns}

    logger.info("Loading TSV file: %s", file_path)
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=dtype_dict,
        nrows=nrows,
        keep_default_na=False,  # Treat 'NA' as actual strings, not NaN
        na_values=["", "NaN", "nan", "NULL", "null", "none"],
        encoding="utf-8",
    )

    validate_columns(df, expected_columns, os.path.basename(file_path))

    # Clean missing values to empty string for safe string manipulations
    for col in expected_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    logger.info("Loaded %d rows from %s", len(df), os.path.basename(file_path))
    return df


def load_source1(
    path_or_root: str, split: str = "train", nrows: Optional[int] = None
) -> pd.DataFrame:
    """Load Source 1 (Reference) records."""
    if os.path.isdir(path_or_root):
        file_path = os.path.join(path_or_root, "dataset", split, f"{split}_source1.tsv")
    else:
        file_path = path_or_root

    return load_tsv_file(file_path, SOURCE_COLUMNS, nrows=nrows)


def load_source2(
    path_or_root: str, split: str = "train", nrows: Optional[int] = None
) -> pd.DataFrame:
    """Load Source 2 (Noisy) records."""
    if os.path.isdir(path_or_root):
        file_path = os.path.join(path_or_root, "dataset", split, f"{split}_source2.tsv")
    else:
        file_path = path_or_root

    return load_tsv_file(file_path, SOURCE_COLUMNS, nrows=nrows)


def load_source3(
    path_or_root: str, split: str = "train", nrows: Optional[int] = None
) -> pd.DataFrame:
    """Load Source 3 (Noisy) records."""
    if os.path.isdir(path_or_root):
        file_path = os.path.join(path_or_root, "dataset", split, f"{split}_source3.tsv")
    else:
        file_path = path_or_root

    return load_tsv_file(file_path, SOURCE_COLUMNS, nrows=nrows)


def load_ground_truth(path_or_root: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load Ground Truth mapping file."""
    if os.path.isdir(path_or_root):
        file_path = os.path.join(
            path_or_root, "dataset", "train", "train_ground_truth.tsv"
        )
    else:
        file_path = path_or_root

    return load_tsv_file(file_path, GROUND_TRUTH_COLUMNS, nrows=nrows)


def parse_ground_truth_mapping(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Convert ground truth dataframe into a dictionary mapping S1 IDs to sets of matched IDs.

    Singletons (S1 records with no match) will map to an empty set.
    """
    mapping: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_str = str(row["matched_entity_ids"]).strip()
        if not matched_str:
            mapping[s1_id] = set()
        else:
            ids = {mid.strip() for mid in matched_str.split(",") if mid.strip()}
            mapping[s1_id] = ids
    return mapping


class DataLoader:
    """Convenience class to manage data loading for train and test splits."""

    def __init__(self, data_root: str):
        self.data_root = data_root

    def load_train_all(self, nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load all four training data files."""
        return {
            "s1": load_source1(self.data_root, split="train", nrows=nrows),
            "s2": load_source2(self.data_root, split="train", nrows=nrows),
            "s3": load_source3(self.data_root, split="train", nrows=nrows),
            "gt": load_ground_truth(self.data_root, nrows=nrows),
        }

    def load_test_all(self, nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load all three test data files with optional row limit."""
        return {
            "s1": load_source1(self.data_root, split="test", nrows=nrows),
            "s2": load_source2(self.data_root, split="test", nrows=nrows),
            "s3": load_source3(self.data_root, split="test", nrows=nrows),
        }
