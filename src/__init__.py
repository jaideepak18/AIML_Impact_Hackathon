"""Business Entity Resolution Package (ML Challenge 2026).

A clean, reproducible, modular architecture for entity resolution across
heterogeneous, noisy business records from multiple sources.
"""

from .blocking import BlockingStats, MultiChannelBlocker
from .config import Config
from .data_loader import (
    DataLoader,
    load_ground_truth,
    load_source1,
    load_source2,
    load_source3,
    parse_ground_truth_mapping,
)
from .evaluation import (
    calculate_entity_metrics,
    comprehensive_evaluation,
    generate_entity_predictions,
    macro_f05_evaluation,
    optimize_threshold,
    run_error_analysis,
)
from .features import FeatureExtractor, sample_candidate_pairs
from .matcher import EntityMatcher, entity_level_train_val_split
from .output import (
    OutputGenerator,
    call_official_validator,
    validate_submission_internal,
)
from .pipeline import run_pipeline
from .preprocessing import (
    Preprocessor,
    extract_address_digits,
    extract_tokens,
    normalize_address,
    normalize_country,
    normalize_name,
    strip_legal_suffixes,
)
from .semantic_model import SemanticModel
from .subset import SubsetStats, build_development_subset

__all__ = [
    "Config",
    "DataLoader",
    "load_source1",
    "load_source2",
    "load_source3",
    "load_ground_truth",
    "parse_ground_truth_mapping",
    "build_development_subset",
    "SubsetStats",
    "Preprocessor",
    "normalize_name",
    "strip_legal_suffixes",
    "normalize_address",
    "extract_address_digits",
    "extract_tokens",
    "normalize_country",
    "MultiChannelBlocker",
    "BlockingStats",
    "FeatureExtractor",
    "sample_candidate_pairs",
    "SemanticModel",
    "EntityMatcher",
    "entity_level_train_val_split",
    "calculate_entity_metrics",
    "macro_f05_evaluation",
    "generate_entity_predictions",
    "optimize_threshold",
    "comprehensive_evaluation",
    "run_error_analysis",
    "OutputGenerator",
    "validate_submission_internal",
    "call_official_validator",
    "run_pipeline",
]
