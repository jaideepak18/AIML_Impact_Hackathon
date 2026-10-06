"""Development subset creation module for Business Entity Resolution.

Extracts a consistent, reproducible development subset (e.g. ~10,000 Source 1 records)
while strictly preserving all true positive Source 2 and Source 3 links,
singletons (unmatched entities), and representative distractors.
"""

from dataclasses import dataclass
import logging
import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from .data_loader import (
    load_ground_truth,
    load_source1,
    load_source2,
    load_source3,
    parse_ground_truth_mapping,
)

logger = logging.getLogger(__name__)


@dataclass
class SubsetStats:
    """Summary statistics for the extracted development subset."""

    n_s1: int
    n_s2: int
    n_s3: int
    n_positive_links: int
    n_singletons: int
    pct_singletons: float
    avg_matches_per_s1: float
    max_matches_per_s1: int
    country_counts: Dict[str, int]

    def report(self) -> str:
        """Formatted diagnostic report."""
        lines = [
            "-" * 55,
            "DEVELOPMENT SUBSET DIAGNOSTIC REPORT",
            "-" * 55,
            f"Source 1 Entities:            {self.n_s1:,}",
            f"Source 2 Records:             {self.n_s2:,}",
            f"Source 3 Records:             {self.n_s3:,}",
            f"Total Positive Ground Truth Links: {self.n_positive_links:,}",
            f"Singleton S1 Entities (0 match):   {self.n_singletons:,} ({self.pct_singletons:.2f}%)",
            f"Average Matches per S1:        {self.avg_matches_per_s1:.3f}",
            f"Maximum Matches for a single S1: {self.max_matches_per_s1}",
            f"Country Distribution (S1):     {dict(self.country_counts)}",
            "-" * 55,
        ]
        return "\n".join(lines)


def filter_target_source_by_ids_and_distractors(
    source_df: pd.DataFrame,
    required_ids: Set[str],
    n_distractors: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Retain 100% of required positive IDs, plus a reproducible sample of distractors."""
    # Split into required positives present in this source vs remaining candidates
    mask_required = source_df["entity_id"].isin(required_ids)
    df_required = source_df[mask_required]

    df_remaining = source_df[~mask_required]

    if n_distractors > 0 and len(df_remaining) > 0:
        n_sample = min(n_distractors, len(df_remaining))
        df_distractors = df_remaining.sample(n=n_sample, random_state=random_seed)
        combined = pd.concat([df_required, df_distractors], ignore_index=True)
    else:
        combined = df_required.copy()

    # Shuffle for uniform distribution
    return combined.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)


def build_development_subset(
    data_root: Optional[str] = None,
    s1_df: Optional[pd.DataFrame] = None,
    s2_df: Optional[pd.DataFrame] = None,
    s3_df: Optional[pd.DataFrame] = None,
    gt_df: Optional[pd.DataFrame] = None,
    n_source1: int = 10000,
    random_seed: int = 42,
    distractor_ratio: float = 3.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, SubsetStats]:
    """Create a consistent, reproducible development subset without mutating original data.

    Steps:
    1. Select n_source1 Source 1 entities using random seed.
    2. Retrieve ground truth for the selected Source 1 entities.
    3. Extract all true matching S2 and S3 entity IDs to guarantee zero positive loss.
    4. Include matching S2/S3 records + sample realistic distractor records.
    5. Maintain all singletons (S1 entities with 0 matches).
    6. Compute and return detailed dataset statistics.

    Returns:
        (s1_subset, s2_subset, s3_subset, gt_subset, stats)
    """
    logger.info("Building development subset (n_source1=%d, seed=%d)", n_source1, random_seed)

    # 1. Load data if not already provided in memory
    if s1_df is None:
        if data_root is None:
            raise ValueError("Either data_root or source DataFrames must be provided.")
        s1_df = load_source1(data_root, split="train")

    if gt_df is None:
        if data_root is None:
            raise ValueError("Either data_root or gt_df must be provided.")
        gt_df = load_ground_truth(data_root)

    # 2. Select Source 1 subset
    total_s1 = len(s1_df)
    if n_source1 > 0 and n_source1 < total_s1:
        s1_subset = s1_df.sample(n=n_source1, random_state=random_seed).copy()
    else:
        s1_subset = s1_df.copy()

    selected_s1_ids: Set[str] = set(s1_subset["entity_id"])

    # 3. Filter ground truth for the selected S1 records
    gt_subset = gt_df[gt_df["source1_entity_id"].isin(selected_s1_ids)].copy()

    # Parse matching targets
    gt_mapping = parse_ground_truth_mapping(gt_subset)
    all_positive_s2_ids: Set[str] = set()
    all_positive_s3_ids: Set[str] = set()
    total_positives = 0
    singletons = 0
    match_counts: List[int] = []

    for s1_id in selected_s1_ids:
        matched = gt_mapping.get(s1_id, set())
        n_m = len(matched)
        match_counts.append(n_m)
        if n_m == 0:
            singletons += 1
        else:
            total_positives += n_m
            for mid in matched:
                if mid.startswith("S2-"):
                    all_positive_s2_ids.add(mid)
                elif mid.startswith("S3-"):
                    all_positive_s3_ids.add(mid)

    # 4. Load & filter Source 2 and Source 3
    # Determine number of distractors
    n_s2_distractors = int(len(all_positive_s2_ids) * distractor_ratio)
    n_s3_distractors = int(len(all_positive_s3_ids) * distractor_ratio)

    if s2_df is None:
        s2_df = load_source2(data_root, split="train")
    if s3_df is None:
        s3_df = load_source3(data_root, split="train")

    s2_subset = filter_target_source_by_ids_and_distractors(
        s2_df, all_positive_s2_ids, n_s2_distractors, random_seed=random_seed
    )
    s3_subset = filter_target_source_by_ids_and_distractors(
        s3_df, all_positive_s3_ids, n_s3_distractors, random_seed=random_seed + 1
    )

    # Country distribution
    country_counts = dict(s1_subset["country"].value_counts())

    stats = SubsetStats(
        n_s1=len(s1_subset),
        n_s2=len(s2_subset),
        n_s3=len(s3_subset),
        n_positive_links=total_positives,
        n_singletons=singletons,
        pct_singletons=(singletons / len(s1_subset) * 100.0) if len(s1_subset) > 0 else 0.0,
        avg_matches_per_s1=float(np.mean(match_counts)) if match_counts else 0.0,
        max_matches_per_s1=max(match_counts) if match_counts else 0,
        country_counts=country_counts,
    )

    if verbose:
        print(stats.report())

    return s1_subset, s2_subset, s3_subset, gt_subset, stats
