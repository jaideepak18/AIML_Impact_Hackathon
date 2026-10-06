"""Feature engineering module for Business Entity Resolution.

Extracts rich lexical, phonetic, structural, and numeric alignment features
for candidate pairs (S1 + S2/S3) with high-efficiency batch processing,
negative sampling, and pure-Python fallbacks.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Attempt high-performance RapidFuzz import with pure-Python fallback
try:
    from rapidfuzz import distance, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    logger.warning("rapidfuzz not installed; falling back to standard library similarity.")


def compute_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Compute Jaccard intersection over union."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def counter_cosine_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute token frequency cosine similarity (TF-IDF / token cosine)."""
    if not tokens1 or not tokens2:
        return 0.0
    from collections import Counter
    import math

    c1 = Counter(tokens1)
    c2 = Counter(tokens2)
    common = set(c1.keys()) & set(c2.keys())
    if not common:
        return 0.0
    dot = sum(c1[t] * c2[t] for t in common)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    return float(dot / (norm1 * norm2)) if (norm1 * norm2) > 0.0 else 0.0


def fallback_ratio(s1: str, s2: str) -> float:
    """Compute basic sequence similarity ratio in range [0, 100]."""
    if not s1 or not s2:
        return 0.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, s1, s2).ratio() * 100.0


def fallback_token_sort_ratio(s1: str, s2: str) -> float:
    """Compute token sort ratio using standard sorting."""
    t1 = " ".join(sorted(s1.split()))
    t2 = " ".join(sorted(s2.split()))
    return fallback_ratio(t1, t2)


def fallback_token_set_ratio(s1: str, s2: str) -> float:
    """Compute token set overlap ratio."""
    tok1 = set(s1.split())
    tok2 = set(s2.split())
    if not tok1 or not tok2:
        return 0.0
    common = " ".join(sorted(tok1 & tok2))
    diff1 = " ".join(sorted(tok1 - tok2))
    diff2 = " ".join(sorted(tok2 - tok1))
    s_common_diff1 = (common + " " + diff1).strip()
    s_common_diff2 = (common + " " + diff2).strip()
    scores = [
        fallback_ratio(common, s_common_diff1),
        fallback_ratio(common, s_common_diff2),
        fallback_ratio(s_common_diff1, s_common_diff2),
    ]
    return max(scores)


def compute_fuzz_features(s1: str, s2: str) -> Tuple[float, float, float, float, float]:
    """Compute (ratio, partial_ratio, token_sort_ratio, token_set_ratio, jaro_winkler).

    All returned values normalized to range [0.0, 1.0].
    """
    if not s1 or not s2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    if HAS_RAPIDFUZZ:
        r = fuzz.ratio(s1, s2) / 100.0
        pr = fuzz.partial_ratio(s1, s2) / 100.0
        tsor = fuzz.token_sort_ratio(s1, s2) / 100.0
        tser = fuzz.token_set_ratio(s1, s2) / 100.0
        jw = distance.JaroWinkler.similarity(s1, s2)
        return r, pr, tsor, tser, jw
    else:
        r = fallback_ratio(s1, s2) / 100.0
        pr = r  # fallback approximation
        tsor = fallback_token_sort_ratio(s1, s2) / 100.0
        tser = fallback_token_set_ratio(s1, s2) / 100.0
        jw = r  # fallback approximation
        return r, pr, tsor, tser, jw


def sample_candidate_pairs(
    cand_df: pd.DataFrame,
    negative_samples_per_positive: int = 10,
    random_seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sample candidate pairs for training to balance the positive/negative ratio.

    Strictly retains 100% of true positive pairs (label == 1), while sampling
    hard and random negative pairs (label == 0) up to the specified ratio.
    """
    if "label" not in cand_df.columns:
        return cand_df

    positives = cand_df[cand_df["label"] == 1]
    negatives = cand_df[cand_df["label"] == 0]

    n_pos = len(positives)
    n_neg = len(negatives)

    if n_pos == 0:
        return cand_df

    target_neg = min(n_neg, n_pos * negative_samples_per_positive)

    # Prioritize hard negatives (low rank / high blocking hits), plus random sample
    half_target = target_neg // 2
    hard_negatives = negatives.sort_values(
        by=["blocking_hits", "candidate_rank"], ascending=[False, True]
    ).head(half_target)

    remaining_negatives = negatives.drop(hard_negatives.index)
    n_random = target_neg - len(hard_negatives)
    random_negatives = (
        remaining_negatives.sample(n=min(n_random, len(remaining_negatives)), random_state=random_seed)
        if n_random > 0 and len(remaining_negatives) > 0
        else pd.DataFrame()
    )

    sampled = pd.concat([positives, hard_negatives, random_negatives], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    if verbose:
        print(
            f"Candidate Sampling: Positives={n_pos:,}, "
            f"Negatives={len(sampled) - n_pos:,} (Ratio: 1:{(len(sampled) - n_pos)/n_pos:.1f})"
        )

    return sampled


class FeatureExtractor:
    """Extracts comprehensive lexical and structural features for candidate pairs."""

    def __init__(self) -> None:
        self.feature_names: List[str] = [
            # Name features
            "name_exact_match",
            "name_stripped_exact_match",
            "name_fuzz_ratio",
            "name_fuzz_partial_ratio",
            "name_token_sort_ratio",
            "name_token_set_ratio",
            "name_jaro_winkler",
            "name_jaccard_tokens",
            "name_common_token_count",
            "name_length_ratio",
            "name_char_3gram_jaccard",
            "name_tfidf_cosine",
            # Address features
            "addr_exact_match",
            "addr_fuzz_ratio",
            "addr_token_sort_ratio",
            "addr_token_set_ratio",
            "addr_jaccard_tokens",
            "addr_common_token_count",
            "addr_numeric_jaccard",
            "addr_exact_digit_match",
            "addr_length_ratio",
            "addr_tfidf_cosine",
            # Cross-field & Structural features
            "country_match",
            "shared_digit_count",
            "has_shared_digits",
            "is_source2",
            "is_source3",
            "blocking_hits",
            "candidate_rank_inv",
            "combined_heuristic_score",
        ]

    def extract_features(
        self,
        candidate_df: pd.DataFrame,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Vectorized/batched computation of pairwise features.

        Uses fast dictionary lookups to avoid slow pandas row operations.
        """
        if verbose:
            logger.info("Extracting features for %d candidate pairs...", len(candidate_df))

        # Index records by entity_id for instant O(1) lookup
        s1_records: Dict[str, Dict[str, Any]] = (
            s1_df.set_index("entity_id")[
                [
                    "normalized_name",
                    "name_without_legal_suffix",
                    "normalized_address",
                    "address_digits",
                    "normalized_country",
                ]
            ].to_dict(orient="index")
        )

        target_records: Dict[str, Dict[str, Any]] = (
            target_df.set_index("entity_id")[
                [
                    "normalized_name",
                    "name_without_legal_suffix",
                    "normalized_address",
                    "address_digits",
                    "normalized_country",
                ]
            ].to_dict(orient="index")
        )

        # Preallocate feature columns as lists for maximum appending speed
        feats: Dict[str, List[float]] = {fname: [] for fname in self.feature_names}

        s1_ids = candidate_df["source1_entity_id"].tolist()
        t_ids = candidate_df["target_entity_id"].tolist()
        hits_list = candidate_df.get("blocking_hits", [1] * len(candidate_df)).tolist()
        ranks_list = candidate_df.get("candidate_rank", [1] * len(candidate_df)).tolist()

        for s1_id, t_id, hits, rank in zip(s1_ids, t_ids, hits_list, ranks_list):
            r1 = s1_records.get(s1_id, {})
            r2 = target_records.get(t_id, {})

            # Retrieve text fields
            n1 = r1.get("normalized_name", "")
            n2 = r2.get("normalized_name", "")
            sn1 = r1.get("name_without_legal_suffix", "")
            sn2 = r2.get("name_without_legal_suffix", "")
            a1 = r1.get("normalized_address", "")
            a2 = r2.get("normalized_address", "")
            d1_str = r1.get("address_digits", "")
            d2_str = r2.get("address_digits", "")
            c1 = r1.get("normalized_country", "UNKNOWN")
            c2 = r2.get("normalized_country", "UNKNOWN")

            # --- Name Features ---
            name_exact = 1.0 if (n1 and n1 == n2) else 0.0
            name_stripped_exact = 1.0 if (sn1 and sn1 == sn2) else 0.0

            r, pr, tsor, tser, jw = compute_fuzz_features(n1, n2)

            t1_set = set(n1.split())
            t2_set = set(n2.split())
            name_jaccard = jaccard_similarity(t1_set, t2_set)
            name_common = float(len(t1_set & t2_set))

            len1, len2 = len(n1), len(n2)
            name_len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 0.0

            ng1 = compute_char_ngrams(n1, 3)
            ng2 = compute_char_ngrams(n2, 3)
            name_char_3g = jaccard_similarity(ng1, ng2)

            # --- Address Features ---
            addr_exact = 1.0 if (a1 and a1 == a2) else 0.0
            ar, _, atsor, atser, _ = compute_fuzz_features(a1, a2)

            at1_set = set(a1.split())
            at2_set = set(a2.split())
            addr_jaccard = jaccard_similarity(at1_set, at2_set)
            addr_common = float(len(at1_set & at2_set))

            alen1, alen2 = len(a1), len(a2)
            addr_len_ratio = (min(alen1, alen2) / max(alen1, alen2)) if max(alen1, alen2) > 0 else 0.0

            # Digits
            d1_set = set(d1_str.split()) if d1_str else set()
            d2_set = set(d2_str.split()) if d2_str else set()
            num_jaccard = jaccard_similarity(d1_set, d2_set)
            exact_digit = 1.0 if (d1_str and d1_str == d2_str) else 0.0
            shared_digits = float(len(d1_set & d2_set))

            # --- Structural & Cross-Field ---
            if c1 != "UNKNOWN" and c2 != "UNKNOWN":
                c_match = 1.0 if c1 == c2 else 0.0
            else:
                c_match = 0.5  # Neutral when country is unspecified

            is_s2 = 1.0 if str(t_id).startswith("S2-") else 0.0
            is_s3 = 1.0 if str(t_id).startswith("S3-") else 0.0

            rank_inv = 1.0 / max(float(rank), 1.0)
            combined_heuristic = 0.60 * tser + 0.40 * atser

            # Append to lists
            feats["name_exact_match"].append(name_exact)
            feats["name_stripped_exact_match"].append(name_stripped_exact)
            feats["name_fuzz_ratio"].append(r)
            feats["name_fuzz_partial_ratio"].append(pr)
            feats["name_token_sort_ratio"].append(tsor)
            feats["name_token_set_ratio"].append(tser)
            feats["name_jaro_winkler"].append(jw)
            feats["name_jaccard_tokens"].append(name_jaccard)
            feats["name_common_token_count"].append(name_common)
            feats["name_length_ratio"].append(name_len_ratio)
            feats["name_char_3gram_jaccard"].append(name_char_3g)
            feats["name_tfidf_cosine"].append(counter_cosine_similarity(n1.split(), n2.split()))

            feats["addr_exact_match"].append(addr_exact)
            feats["addr_fuzz_ratio"].append(ar)
            feats["addr_token_sort_ratio"].append(atsor)
            feats["addr_token_set_ratio"].append(atser)
            feats["addr_jaccard_tokens"].append(addr_jaccard)
            feats["addr_common_token_count"].append(addr_common)
            feats["addr_numeric_jaccard"].append(num_jaccard)
            feats["addr_exact_digit_match"].append(exact_digit)
            feats["addr_length_ratio"].append(addr_len_ratio)
            feats["addr_tfidf_cosine"].append(counter_cosine_similarity(a1.split(), a2.split()))

            feats["country_match"].append(c_match)
            feats["shared_digit_count"].append(shared_digits)
            feats["has_shared_digits"].append(1.0 if shared_digits > 0 else 0.0)
            feats["is_source2"].append(is_s2)
            feats["is_source3"].append(is_s3)
            feats["blocking_hits"].append(float(hits))
            feats["candidate_rank_inv"].append(rank_inv)
            feats["combined_heuristic_score"].append(combined_heuristic)

        feature_df = pd.DataFrame(feats)
        # Ensure identical row index with candidate_df
        feature_df.index = candidate_df.index

        return feature_df
