"""Configuration module for the Business Entity Resolution pipeline.

Provides a centralized, reproducible configuration dataclass with defaults
customized for Google Colab execution on the ML Challenge 2026 dataset.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, Optional


@dataclass
class Config:
    """Central configuration for business entity resolution pipeline."""

    # -------------------------------------------------------------------------
    # Path & Environment Configuration
    # -------------------------------------------------------------------------
    # Path to dataset root directory containing dataset/train and dataset/test
    data_root: str = field(
        default_factory=lambda: (
            os.environ.get("DATA_ROOT")
            if os.environ.get("DATA_ROOT") and os.path.exists(os.environ.get("DATA_ROOT"))
            else "/content/student_resource"
            if os.path.exists("/content/student_resource/dataset")
            else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        )
    )

    # Output directory where matching_results.tsv and candidate_pairs.tsv are saved
    output_dir: Optional[str] = None

    # Model checkpoint save/load path
    model_save_path: Optional[str] = None

    # -------------------------------------------------------------------------
    # Dataset Subsetting & Sampling
    # -------------------------------------------------------------------------
    # Number of Source 1 entities for the development set (-1 to use all)
    n_source1: int = 10000

    # Optional row limits for data loading (useful for quick local runs / sample outputs)
    max_train_rows: Optional[int] = None
    max_test_rows: Optional[int] = None

    # Reproducibility seed
    random_seed: int = 42

    # Validation split fraction (split at Source 1 entity level)
    validation_size: float = 0.2

    # Ratio of negative distractor records to retain per positive during training
    negative_samples_per_positive: int = 10

    # -------------------------------------------------------------------------
    # Blocking & Candidate Generation
    # -------------------------------------------------------------------------
    # Maximum candidate records (from S2/S3) to consider per Source 1 entity
    max_candidates_per_s1: int = 75

    # Minimum character length for tokens used in indexing
    min_token_len: int = 3

    # Maximum records allowed in a single inverted index bucket before pruning
    max_bucket_size: int = 5000

    # Whether to enforce country matching when both records specify a country
    # Countries are treated as an open set of string labels (e.g. US, India, France)
    filter_by_country: bool = True

    # -------------------------------------------------------------------------
    # Feature Engineering & Semantic Model
    # -------------------------------------------------------------------------
    # Whether to compute semantic sentence transformer embeddings
    use_semantic: bool = False

    # License-compliant transformer model (Apache 2.0, ~22M params)
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Max sequence length for semantic embedding text
    max_sequence_length: int = 128

    # Batch size for embedding inference and feature calculation
    batch_size: int = 256

    # -------------------------------------------------------------------------
    # Hybrid Matching Model & Decision Engine
    # -------------------------------------------------------------------------
    # Classifier algorithm: "hist_gradient_boosting", "random_forest", or "logistic_regression"
    classifier_type: str = "hist_gradient_boosting"

    # Default match probability threshold (dynamically tuned on validation set)
    match_threshold: float = 0.50

    # Threshold search grid for macro F0.5 optimization
    threshold_min: float = 0.10
    threshold_max: float = 0.95
    threshold_step: float = 0.05

    # Secondary match margin: candidates within this margin of top candidate score
    # can be considered if above match_threshold
    allow_multiple_matches: bool = True

    # -------------------------------------------------------------------------
    # Runtime & Logging
    # -------------------------------------------------------------------------
    verbose: bool = True
    n_jobs: int = -1

    def __post_init__(self) -> None:
        """Resolve dynamic paths and ensure output directory existence."""
        if self.output_dir is None:
            self.output_dir = os.path.join(self.data_root, "output")
        if self.model_save_path is None:
            self.model_save_path = os.path.join(self.output_dir, "entity_matcher.pkl")
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def train_s1_path(self) -> str:
        return os.path.join(self.data_root, "dataset", "train", "train_source1.tsv")

    @property
    def train_s2_path(self) -> str:
        return os.path.join(self.data_root, "dataset", "train", "train_source2.tsv")

    @property
    def train_s3_path(self) -> str:
        return os.path.join(self.data_root, "dataset", "train", "train_source3.tsv")

    @property
    def train_gt_path(self) -> str:
        return os.path.join(self.data_root, "dataset", "train", "train_ground_truth.tsv")

    @property
    def test_s1_path(self) -> str:
        return os.path.join(self.data_root, "dataset", "test", "test_source1.tsv")

    @property
    def test_s2_path(self) -> str:
        return os.path.join(self.data_root, "dataset", "test", "test_source2.tsv")

    @property
    def test_s3_path(self) -> str:
        return os.path.join(self.data_root, "dataset", "test", "test_source3.tsv")

    @property
    def test_dir(self) -> str:
        return os.path.join(self.data_root, "dataset", "test")

    @property
    def matching_output_path(self) -> str:
        return os.path.join(self.output_dir, "matching_results.tsv")

    @property
    def candidate_output_path(self) -> str:
        return os.path.join(self.output_dir, "candidate_pairs.tsv")

    @property
    def validator_script_path(self) -> str:
        return os.path.join(self.data_root, "utils", "validate_submission.py")

    def summary(self) -> str:
        """Return a formatted string of the current configuration parameters."""
        lines = [
            "=" * 60,
            "BUSINESS ENTITY RESOLUTION PIPELINE CONFIGURATION",
            "=" * 60,
            f"DATA_ROOT:                      {self.data_root}",
            f"OUTPUT_DIR:                     {self.output_dir}",
            f"N_SOURCE1 (Dev Subset):         {self.n_source1}",
            f"RANDOM_SEED:                    {self.random_seed}",
            f"VALIDATION_SIZE:                {self.validation_size}",
            f"MAX_CANDIDATES_PER_S1:          {self.max_candidates_per_s1}",
            f"NEGATIVE_SAMPLES_PER_POSITIVE:  {self.negative_samples_per_positive}",
            f"CLASSIFIER_TYPE:                {self.classifier_type}",
            f"USE_SEMANTIC:                   {self.use_semantic}",
            f"MODEL_NAME:                     {self.model_name}",
            f"MATCH_THRESHOLD (Initial):       {self.match_threshold}",
            f"ALLOW_MULTIPLE_MATCHES:         {self.allow_multiple_matches}",
            f"FILTER_BY_COUNTRY:              {self.filter_by_country} (open set)",
            "=" * 60,
        ]
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Instantiate Config from a dictionary of overrides."""
        return cls(**config_dict)
