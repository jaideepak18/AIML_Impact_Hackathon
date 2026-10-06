"""Multi-channel inverted-index blocking module for Business Entity Resolution.

Constructs high-recall indexed candidate pools across Source 2 and Source 3 without
performing Cartesian cross-products, maintaining high positive pair recall
and computing diagnostic coverage statistics (S2 recall, S3 recall, reduction ratio).
"""

from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from .preprocessing import extract_address_digits, extract_tokens

logger = logging.getLogger(__name__)


@dataclass
class BlockingStats:
    """Summary metrics of candidate blocking performance."""

    n_s1: int
    n_target_records: int
    total_candidate_pairs: int
    avg_candidates_per_s1: float
    median_candidates_per_s1: float
    min_candidates: int
    max_candidates: int
    n_s1_zero_candidates: int
    reduction_ratio: float
    blocking_recall: Optional[float] = None
    true_positives_captured: Optional[int] = None
    total_ground_truth_positives: Optional[int] = None
    missed_true_links: Optional[int] = None
    s2_blocking_recall: Optional[float] = None
    s2_captured: Optional[int] = None
    s2_total: Optional[int] = None
    s3_blocking_recall: Optional[float] = None
    s3_captured: Optional[int] = None
    s3_total: Optional[int] = None

    def report(self) -> str:
        """Formatted string diagnostic report adhering to official blocking evaluation."""
        lines = [
            "-" * 55,
            "Blocking Evaluation",
            "-" * 55,
        ]
        if self.total_ground_truth_positives is not None:
            missed = self.missed_true_links if self.missed_true_links is not None else 0
            captured = self.true_positives_captured if self.true_positives_captured is not None else 0
            recall_pct = (self.blocking_recall * 100.0) if self.blocking_recall is not None else 0.0
            lines.extend([
                f"Total true links:              {self.total_ground_truth_positives:,}",
                f"True links in candidate set:   {captured:,}",
                f"Missed true links:             {missed:,}",
                f"Blocking Recall:               {recall_pct:.2f}%",
            ])
            if self.s2_total is not None and self.s2_total > 0:
                s2_rec = (self.s2_blocking_recall * 100.0) if self.s2_blocking_recall is not None else 0.0
                lines.append(f"  - S2 Blocking Recall:        {s2_rec:.2f}% ({self.s2_captured:,} / {self.s2_total:,})")
            if self.s3_total is not None and self.s3_total > 0:
                s3_rec = (self.s3_blocking_recall * 100.0) if self.s3_blocking_recall is not None else 0.0
                lines.append(f"  - S3 Blocking Recall:        {s3_rec:.2f}% ({self.s3_captured:,} / {self.s3_total:,})")
            lines.append("")

        lines.extend([
            f"Total Candidate Pairs:         {self.total_candidate_pairs:,}",
            f"Average Candidates/S1:         {self.avg_candidates_per_s1:.2f}",
            f"Median Candidates/S1:          {int(round(self.median_candidates_per_s1))}",
            f"Maximum Candidates/S1:         {self.max_candidates}",
            f"Reduction Ratio:               {self.reduction_ratio * 100.0:.4f}%",
            "-" * 55,
        ])
        return "\n".join(lines)


def get_record_blocking_keys(
    norm_name: str,
    stripped_name: str,
    norm_addr: str,
    addr_digits: str,
    country: str,
    min_token_len: int = 3,
) -> Set[str]:
    """Generate multi-channel blocking keys across 9 complementary channels.

    Channels:
    - CHANNEL A: First distinctive normalized business-name token.
    - CHANNEL B: First + second normalized name tokens (bigram).
    - CHANNEL C: Name token + address number.
    - CHANNEL D: Distinctive address number (>= 2 digits).
    - CHANNEL E: Character 4-gram / prefix signature for business name.
    - CHANNEL F: Country + each distinctive name token.
    - CHANNEL G: Country + address street token.
    - CHANNEL H: Legal-suffix-stripped exact normalized name.
    - CHANNEL I: Sorted first two tokens (order-invariant bi-gram).

    Open-set country labels (US, India, France, etc.) are respected,
    with selective cross-country fallback channels for high-specificity keys.
    """
    keys: Set[str] = set()

    # Normalize country partition (supports open set)
    c_part = country.strip().upper() if country else "ALL"

    # Extract name tokens (clean, non-empty, above minimum token length)
    name_tokens = extract_tokens(stripped_name or norm_name, remove_stopwords=True)
    clean_name_tokens = [t for t in name_tokens if len(t) >= min_token_len]

    # Extract address tokens (street names, etc.)
    addr_tokens = extract_tokens(norm_addr, remove_stopwords=True)
    clean_addr_tokens = [t for t in addr_tokens if len(t) >= min_token_len and not t.isdigit()]

    # Extract address numbers
    nums = addr_digits.split() if addr_digits else []
    clean_nums = [n for n in nums if len(n) >= 2]

    # CHANNEL A: First distinctive normalized business-name token
    if clean_name_tokens:
        tok1 = clean_name_tokens[0]
        keys.add(f"cA_{c_part}_{tok1}")

    # CHANNEL B: First + second normalized name tokens
    if len(clean_name_tokens) >= 2:
        tok_bi = f"{clean_name_tokens[0]}_{clean_name_tokens[1]}"
        keys.add(f"cB_{c_part}_{tok_bi}")
        keys.add(f"cB_ALL_{tok_bi}")

    # CHANNEL C: Name token + address number
    if clean_name_tokens and clean_nums:
        for tok in clean_name_tokens[:2]:
            for num in clean_nums[:2]:
                keys.add(f"cC_{c_part}_{tok}_{num}")
                keys.add(f"cC_ALL_{tok}_{num}")

    # CHANNEL D: Distinctive address number (>= 2 digits)
    if clean_nums:
        for num in clean_nums[:2]:
            keys.add(f"cD_{c_part}_{num}")

    # CHANNEL E: Character 4-gram / signature for business name
    compact_name = (stripped_name or norm_name).replace(" ", "")
    if len(compact_name) >= 4:
        keys.add(f"cE_{c_part}_{compact_name[:4]}")
        if len(compact_name) >= 8:
            keys.add(f"cE_{c_part}_{compact_name[:4]}_{compact_name[-4:]}")

    # CHANNEL F: Country + distinctive name tokens (all distinctive tokens up to 3)
    for tok in clean_name_tokens[:3]:
        keys.add(f"cF_{c_part}_{tok}")

    # CHANNEL G: Country + address street token
    for atok in clean_addr_tokens[:2]:
        keys.add(f"cG_{c_part}_{atok}")

    # CHANNEL H: Legal-suffix-stripped full name
    compact_stripped = " ".join(clean_name_tokens)
    if compact_stripped:
        keys.add(f"cH_{c_part}_{compact_stripped}")
        keys.add(f"cH_ALL_{compact_stripped}")

    # CHANNEL I: Sorted first two tokens (order-invariant bi-gram)
    if len(clean_name_tokens) >= 2:
        t1, t2 = sorted([clean_name_tokens[0], clean_name_tokens[1]])
        keys.add(f"cI_{c_part}_{t1}_{t2}")
        keys.add(f"cI_ALL_{t1}_{t2}")

    return keys


class MultiChannelBlocker:
    """Inverted index candidate generation engine across multiple complementary channels."""

    def __init__(
        self,
        max_candidates_per_s1: int = 75,
        min_token_len: int = 3,
        max_bucket_size: int = 5000,
        filter_by_country: bool = True,
    ) -> None:
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.min_token_len = min_token_len
        self.max_bucket_size = max_bucket_size
        self.filter_by_country = filter_by_country
        self.inverted_index: Dict[str, List[str]] = defaultdict(list)
        self.target_record_count: int = 0

    def fit(self, target_df: pd.DataFrame) -> "MultiChannelBlocker":
        """Build inverted index over target records (Source 2 and/or Source 3)."""
        logger.info("Fitting inverted index on %d target records...", len(target_df))
        self.inverted_index.clear()
        self.target_record_count = len(target_df)

        for _, row in target_df.iterrows():
            target_id = str(row["entity_id"]).strip()
            norm_name = str(row.get("normalized_name", ""))
            stripped_name = str(row.get("name_without_legal_suffix", ""))
            norm_addr = str(row.get("normalized_address", ""))
            addr_digits = str(row.get("address_digits", ""))
            country = str(row.get("normalized_country", ""))

            keys = get_record_blocking_keys(
                norm_name=norm_name,
                stripped_name=stripped_name,
                norm_addr=norm_addr,
                addr_digits=addr_digits,
                country=country if self.filter_by_country else "ALL",
                min_token_len=self.min_token_len,
            )

            for key in keys:
                self.inverted_index[key].append(target_id)

        # Prune excessively large buckets to prevent Cartesian explosions on frequent stopwords
        pruned_keys = 0
        for key in list(self.inverted_index.keys()):
            if len(self.inverted_index[key]) > self.max_bucket_size:
                del self.inverted_index[key]
                pruned_keys += 1

        logger.info(
            "Inverted index built: %d unique keys (%d pruned due to bucket cap > %d)",
            len(self.inverted_index),
            pruned_keys,
            self.max_bucket_size,
        )
        return self

    def query_candidates(
        self, s1_df: pd.DataFrame
    ) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
        """Generate candidate target IDs for each Source 1 record.

        Returns:
            candidate_mapping: Dict[s1_id, list of top target_ids]
            hit_counts_mapping: Dict[s1_id, Dict[target_id, hit_count]]
        """
        logger.info("Querying candidates for %d Source 1 records...", len(s1_df))
        candidate_mapping: Dict[str, List[str]] = {}
        hit_counts_mapping: Dict[str, Dict[str, int]] = {}

        for _, row in s1_df.iterrows():
            s1_id = str(row["entity_id"]).strip()
            norm_name = str(row.get("normalized_name", ""))
            stripped_name = str(row.get("name_without_legal_suffix", ""))
            norm_addr = str(row.get("normalized_address", ""))
            addr_digits = str(row.get("address_digits", ""))
            country = str(row.get("normalized_country", ""))

            keys = get_record_blocking_keys(
                norm_name=norm_name,
                stripped_name=stripped_name,
                norm_addr=norm_addr,
                addr_digits=addr_digits,
                country=country if self.filter_by_country else "ALL",
                min_token_len=self.min_token_len,
            )

            # Accumulate channel hits across all matching keys
            hit_counter: Dict[str, int] = defaultdict(int)
            for key in keys:
                target_ids = self.inverted_index.get(key, [])
                for tid in target_ids:
                    hit_counter[tid] += 1

            if not hit_counter:
                candidate_mapping[s1_id] = []
                hit_counts_mapping[s1_id] = {}
                continue

            # Rank candidates: prioritize candidates hit by multiple channels
            ranked = sorted(
                hit_counter.items(), key=lambda item: item[1], reverse=True
            )

            top_candidates = [tid for tid, _ in ranked[: self.max_candidates_per_s1]]
            candidate_mapping[s1_id] = top_candidates
            hit_counts_mapping[s1_id] = {
                tid: hits for tid, hits in ranked[: self.max_candidates_per_s1]
            }

        return candidate_mapping, hit_counts_mapping

    def generate_candidate_dataframe(
        self,
        s1_df: pd.DataFrame,
        candidate_mapping: Dict[str, List[str]],
        hit_counts_mapping: Dict[str, Dict[str, int]],
        ground_truth_mapping: Optional[Dict[str, Set[str]]] = None,
    ) -> Tuple[pd.DataFrame, BlockingStats]:
        """Convert candidate mappings into a tabular candidate-pair DataFrame with diagnostics."""
        rows = []
        counts: List[int] = []
        zero_candidates = 0

        # Detailed ground truth tracking for S2 vs S3 recall
        true_captured = 0
        total_gt_positives = 0
        s2_captured = 0
        s2_total = 0
        s3_captured = 0
        s3_total = 0

        if ground_truth_mapping is not None:
            for s1_id in s1_df["entity_id"]:
                gt_matches = ground_truth_mapping.get(str(s1_id).strip(), set())
                total_gt_positives += len(gt_matches)
                for mid in gt_matches:
                    if mid.startswith("S2-"):
                        s2_total += 1
                    elif mid.startswith("S3-"):
                        s3_total += 1

        for s1_id in s1_df["entity_id"]:
            s1_id_str = str(s1_id).strip()
            cands = candidate_mapping.get(s1_id_str, [])
            n_cands = len(cands)
            counts.append(n_cands)

            if n_cands == 0:
                zero_candidates += 1
                continue

            gt_matches = (
                ground_truth_mapping.get(s1_id_str, set())
                if ground_truth_mapping is not None
                else set()
            )

            hits_dict = hit_counts_mapping.get(s1_id_str, {})
            for rank_idx, target_id in enumerate(cands, start=1):
                is_match = 1 if target_id in gt_matches else 0
                if is_match == 1:
                    true_captured += 1
                    if target_id.startswith("S2-"):
                        s2_captured += 1
                    elif target_id.startswith("S3-"):
                        s3_captured += 1

                rows.append({
                    "source1_entity_id": s1_id_str,
                    "target_entity_id": target_id,
                    "blocking_hits": hits_dict.get(target_id, 1),
                    "candidate_rank": rank_idx,
                    "label": is_match,
                })

        cand_df = pd.DataFrame(rows)
        total_pairs = len(cand_df)
        n_s1 = len(s1_df)

        max_possible_pairs = n_s1 * self.target_record_count
        reduction_ratio = (
            1.0 - (total_pairs / max_possible_pairs)
            if max_possible_pairs > 0
            else 1.0
        )

        recall = (
            (true_captured / total_gt_positives)
            if total_gt_positives > 0
            else None
        )
        s2_recall = (
            (s2_captured / s2_total)
            if s2_total > 0
            else None
        )
        s3_recall = (
            (s3_captured / s3_total)
            if s3_total > 0
            else None
        )
        missed = (
            (total_gt_positives - true_captured)
            if total_gt_positives > 0
            else None
        )

        stats = BlockingStats(
            n_s1=n_s1,
            n_target_records=self.target_record_count,
            total_candidate_pairs=total_pairs,
            avg_candidates_per_s1=float(np.mean(counts)) if counts else 0.0,
            median_candidates_per_s1=float(np.median(counts)) if counts else 0.0,
            min_candidates=min(counts) if counts else 0,
            max_candidates=max(counts) if counts else 0,
            n_s1_zero_candidates=zero_candidates,
            reduction_ratio=reduction_ratio,
            blocking_recall=recall,
            true_positives_captured=true_captured if ground_truth_mapping else None,
            total_ground_truth_positives=total_gt_positives if ground_truth_mapping else None,
            missed_true_links=missed,
            s2_blocking_recall=s2_recall,
            s2_captured=s2_captured if ground_truth_mapping else None,
            s2_total=s2_total if ground_truth_mapping else None,
            s3_blocking_recall=s3_recall,
            s3_captured=s3_captured if ground_truth_mapping else None,
            s3_total=s3_total if ground_truth_mapping else None,
        )

        return cand_df, stats
