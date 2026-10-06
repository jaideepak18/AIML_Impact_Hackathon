"""Output generation and submission validation module for Business Entity Resolution.

Writes strictly formatted tab-separated output files (matching_results.tsv and candidate_pairs.tsv),
enforces structural constraints (uniqueness, S2/S3 prefixes, subset verification),
and provides execution interfaces for the official validation utility.
"""

import logging
import os
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


class OutputGenerator:
    """Formats and writes submission TSV files according to ML Challenge 2026 specifications."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_id_list_tsv(
        self,
        filepath: str,
        header: List[str],
        id_mapping: Dict[str, Iterable[str]],
        ordered_s1_ids: List[str],
    ) -> str:
        """Write tab-separated file with one row per Source 1 entity and comma-separated target IDs.

        Guarantees:
        - UTF-8 encoding
        - Explicit tab delimiter
        - Empty target string for singletons (followed by newline)
        - No duplicate target IDs
        - Exactly one row per entity in ordered_s1_ids
        """
        logger.info("Writing TSV output: %s (%d rows)...", filepath, len(ordered_s1_ids))

        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            # Write header
            f.write(f"{header[0]}{DELIM}{header[1]}\n")

            for s1_id in ordered_s1_ids:
                s1_clean = str(s1_id).strip()
                raw_targets = id_mapping.get(s1_clean, [])

                # Deduplicate while preserving deterministic order
                seen = set()
                deduped = []
                for tid in raw_targets:
                    tid_clean = str(tid).strip()
                    if tid_clean and tid_clean not in seen:
                        # Safety check: exclude self-matches or malformed prefixes
                        if tid_clean.startswith(("S2-", "S3-")):
                            seen.add(tid_clean)
                            deduped.append(tid_clean)

                target_str = ",".join(deduped)
                f.write(f"{s1_clean}{DELIM}{target_str}\n")

        return filepath

    def generate_matching_results(
        self,
        predictions_mapping: Dict[str, Set[str]],
        ordered_s1_ids: List[str],
        filename: str = "matching_results.tsv",
    ) -> str:
        """Generate official leaderboard submission file: matching_results.tsv."""
        filepath = os.path.join(self.output_dir, filename)
        return self.write_id_list_tsv(
            filepath=filepath,
            header=MATCHING_HEADER,
            id_mapping=predictions_mapping,
            ordered_s1_ids=ordered_s1_ids,
        )

    def generate_candidate_pairs(
        self,
        candidate_mapping: Dict[str, List[str]],
        ordered_s1_ids: List[str],
        filename: str = "candidate_pairs.tsv",
    ) -> str:
        """Generate blocking candidate set file: candidate_pairs.tsv."""
        filepath = os.path.join(self.output_dir, filename)
        return self.write_id_list_tsv(
            filepath=filepath,
            header=CANDIDATE_HEADER,
            id_mapping=candidate_mapping,
            ordered_s1_ids=ordered_s1_ids,
        )


def validate_submission_internal(
    matching_path: str,
    candidate_path: Optional[str],
    test_source1_path: str,
    verbose: bool = True,
) -> Tuple[bool, List[str], List[str]]:
    """Perform fast, standalone internal validation checks prior to external submission."""
    errors: List[str] = []
    warnings: List[str] = []

    if not os.path.isfile(matching_path):
        errors.append(f"Matching results file not found at: {matching_path}")
        return False, errors, warnings

    if not os.path.isfile(test_source1_path):
        errors.append(f"Required test_source1.tsv not found at: {test_source1_path}")
        return False, errors, warnings

    # Load required Source 1 IDs
    with open(test_source1_path, "r", encoding="utf-8") as f:
        next(f, None)
        required_s1 = {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}

    # Parse matching results
    seen_s1 = set()
    matched_map: Dict[str, Set[str]] = {}

    with open(matching_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(DELIM)
        if header != MATCHING_HEADER:
            errors.append(
                f"matching_results.tsv: Unexpected header {header}. Expected {MATCHING_HEADER}"
            )

        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split(DELIM)
            if len(parts) < 2:
                errors.append(f"Line {line_num} in matching_results.tsv malformed: {line!r}")
                continue
            s1_id = parts[0].strip()
            rest = parts[1].strip()

            if s1_id in seen_s1:
                errors.append(f"Duplicate Source 1 ID found: {s1_id} at line {line_num}")
            seen_s1.add(s1_id)

            m_ids = [m.strip() for m in rest.split(",") if m.strip()]
            if len(m_ids) != len(set(m_ids)):
                errors.append(f"Duplicate matched IDs in list for S1: {s1_id}")

            for m in m_ids:
                if m.startswith("S1-"):
                    errors.append(f"Invalid self-match to Source 1: {m} in row {s1_id}")
                elif not m.startswith(("S2-", "S3-")):
                    errors.append(f"Invalid entity ID prefix: {m} in row {s1_id}")

            matched_map[s1_id] = set(m_ids)

    # Check completeness
    missing_s1 = required_s1 - seen_s1
    if missing_s1:
        errors.append(f"{len(missing_s1)} required S1 entities missing from matching_results.tsv")

    extra_s1 = seen_s1 - required_s1
    if extra_s1:
        errors.append(f"{len(extra_s1)} unknown S1 IDs present in matching_results.tsv")

    # Candidate pairs cross-verification (if provided)
    if candidate_path and os.path.isfile(candidate_path):
        candidate_map: Dict[str, Set[str]] = {}
        with open(candidate_path, "r", encoding="utf-8") as f:
            c_header = f.readline().rstrip("\n").split(DELIM)
            if c_header != CANDIDATE_HEADER:
                errors.append(f"candidate_pairs.tsv: Unexpected header {c_header}")

            for line in f:
                parts = line.rstrip("\n").split(DELIM)
                if len(parts) >= 2:
                    s1 = parts[0].strip()
                    cands = {c.strip() for c in parts[1].split(",") if c.strip()}
                    candidate_map[s1] = cands

        # Verify that all matches are a subset of candidates
        not_in_cands = 0
        for s1_id, m_set in matched_map.items():
            c_set = candidate_map.get(s1_id, set())
            diff = m_set - c_set
            if diff:
                not_in_cands += len(diff)
        if not_in_cands > 0:
            warnings.append(
                f"{not_in_cands} matched entity IDs were not present in candidate_pairs.tsv"
            )

    passed = len(errors) == 0

    if verbose:
        print("\n" + "=" * 55)
        print("INTERNAL SUBMISSION VALIDATION REPORT")
        print("=" * 55)
        print(f"Status:             {'PASS (Safe to Submit)' if passed else 'FAIL'}")
        print(f"Matching File:      {matching_path}")
        print(f"Total Rows:         {len(seen_s1):,} / {len(required_s1):,} required")
        if warnings:
            for w in warnings:
                print(f"WARNING:            {w}")
        if errors:
            print("\nErrors Found:")
            for i, err in enumerate(errors[:10], 1):
                print(f"  {i}. {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors.")
        print("=" * 55 + "\n")

    return passed, errors, warnings


def call_official_validator(
    validator_script_path: str,
    matching_path: str,
    candidate_path: Optional[str] = None,
    test_dir: Optional[str] = None,
    check_ids: bool = False,
) -> int:
    """Execute the official utils/validate_submission.py script via subprocess."""
    if not os.path.isfile(validator_script_path):
        logger.warning("Official validator script not found at %s", validator_script_path)
        return -1

    cmd = [
        sys.executable,
        validator_script_path,
        "--matching",
        matching_path,
    ]
    if candidate_path:
        cmd.extend(["--candidate", candidate_path])
    if test_dir:
        cmd.extend(["--test-dir", test_dir])
    if check_ids:
        cmd.append("--check-ids")

    print(f"\nRunning official validator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode
