"""Master end-to-end pipeline orchestrator for Business Entity Resolution.

Coordinates data loading, development subset creation, text normalization,
multi-channel blocking, feature extraction, hybrid model training,
validation macro F0.5 threshold optimization, test set inference,
and strict submission output formatting and validation.
"""

import argparse
import logging
import os
import random
import sys
import time
from typing import Optional, Set
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .blocking import MultiChannelBlocker
from .config import Config
from .data_loader import DataLoader, parse_ground_truth_mapping
from .evaluation import (
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
from .preprocessing import Preprocessor
from .semantic_model import SemanticModel
from .subset import build_development_subset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Pipeline")


def set_reproducibility(seed: int = 42) -> None:
    """Set global random seeds for reproducible execution across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


class PipelineTimer:
    """Stopwatch utility to report execution elapsed times for each major stage."""

    def __init__(self) -> None:
        self.stage_start: float = time.time()
        self.pipeline_start: float = time.time()

    def start_stage(self, stage_name: str) -> None:
        print("\n" + "=" * 70)
        print(f"STAGE: {stage_name}")
        print("=" * 70)
        self.stage_start = time.time()

    def end_stage(self, stage_name: str) -> None:
        elapsed = time.time() - self.stage_start
        print(f">> Completed: {stage_name} in {elapsed:.2f}s")

    def total_time(self) -> float:
        return time.time() - self.pipeline_start


def run_pipeline(
    config: Optional[Config] = None,
    run_test_inference: bool = True,
) -> None:
    """Execute the complete entity resolution pipeline."""
    if config is None:
        config = Config()

    print(config.summary())
    set_reproducibility(config.random_seed)
    timer = PipelineTimer()

    # -------------------------------------------------------------------------
    # 1. DATA LOADING & SUBSET CREATION
    # -------------------------------------------------------------------------
    timer.start_stage("1. Data Loading & Development Subset Selection")
    loader = DataLoader(config.data_root)

    train_data = loader.load_train_all(nrows=config.max_train_rows)
    s1_full = train_data["s1"]
    s2_full = train_data["s2"]
    s3_full = train_data["s3"]
    gt_full = train_data["gt"]

    # Build consistent development subset (N_SOURCE1=10000 by default)
    s1_dev, s2_dev, s3_dev, gt_dev, subset_stats = build_development_subset(
        s1_df=s1_full,
        s2_df=s2_full,
        s3_df=s3_full,
        gt_df=gt_full,
        n_source1=config.n_source1,
        random_seed=config.random_seed,
        distractor_ratio=float(config.negative_samples_per_positive) / 2.0,
        verbose=config.verbose,
    )
    gt_mapping = parse_ground_truth_mapping(gt_dev)
    timer.end_stage("1. Data Loading & Development Subset Selection")

    # -------------------------------------------------------------------------
    # 2. PREPROCESSING & TEXT NORMALIZATION
    # -------------------------------------------------------------------------
    timer.start_stage("2. Text Normalization & Legal Suffix Cleaning")
    preprocessor = Preprocessor()

    s1_clean = preprocessor.process_dataframe(s1_dev)
    s2_clean = preprocessor.process_dataframe(s2_dev)
    s3_clean = preprocessor.process_dataframe(s3_dev)

    # Combine target sources into unified target pool
    target_clean = pd.concat([s2_clean, s3_clean], ignore_index=True)
    logger.info("Total target records (S2 + S3): %d", len(target_clean))
    timer.end_stage("2. Text Normalization & Legal Suffix Cleaning")

    # -------------------------------------------------------------------------
    # 3. MULTI-CHANNEL BLOCKING
    # -------------------------------------------------------------------------
    timer.start_stage("3. Multi-Channel Inverted Index Candidate Generation")
    blocker = MultiChannelBlocker(
        max_candidates_per_s1=config.max_candidates_per_s1,
        min_token_len=config.min_token_len,
        filter_by_country=config.filter_by_country,
    )
    blocker.fit(target_clean)

    candidate_mapping, hit_counts_mapping = blocker.query_candidates(s1_clean)
    candidate_df, blocking_stats = blocker.generate_candidate_dataframe(
        s1_df=s1_clean,
        candidate_mapping=candidate_mapping,
        hit_counts_mapping=hit_counts_mapping,
        ground_truth_mapping=gt_mapping,
    )
    print(blocking_stats.report())
    timer.end_stage("3. Multi-Channel Inverted Index Candidate Generation")

    # -------------------------------------------------------------------------
    # 4. CANDIDATE SAMPLING & FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    timer.start_stage("4. Feature Engineering (Lexical & Structural)")
    # Sample negative pairs for balanced model training
    train_candidate_pairs = sample_candidate_pairs(
        candidate_df,
        negative_samples_per_positive=config.negative_samples_per_positive,
        random_seed=config.random_seed,
        verbose=config.verbose,
    )

    feature_extractor = FeatureExtractor()
    X_features = feature_extractor.extract_features(
        candidate_df=train_candidate_pairs,
        s1_df=s1_clean,
        target_df=target_clean,
        verbose=config.verbose,
    )

    # Optional Semantic Embeddings
    if config.use_semantic:
        timer.start_stage("4b. Semantic Similarity Embeddings")
        semantic_model = SemanticModel(
            model_name=config.model_name,
            batch_size=config.batch_size,
            max_sequence_length=config.max_sequence_length,
        )
        if semantic_model.is_available:
            X_sem = semantic_model.extract_semantic_features(
                candidate_df=train_candidate_pairs,
                s1_df=s1_clean,
                target_df=target_clean,
                verbose=config.verbose,
            )
            X_features = pd.concat([X_features, X_sem], axis=1)

    y_train_full = train_candidate_pairs["label"].values
    timer.end_stage("4. Feature Engineering (Lexical & Structural)")

    # -------------------------------------------------------------------------
    # 5. ENTITY-LEVEL TRAIN / VALIDATION SPLIT
    # -------------------------------------------------------------------------
    timer.start_stage("5. Entity-Level Data Split (Zero Leakage)")
    train_pairs_df, val_pairs_df, train_s1_ids, val_s1_ids = entity_level_train_val_split(
        all_s1_ids=s1_dev["entity_id"],
        candidate_df=train_candidate_pairs,
        validation_size=config.validation_size,
        random_seed=config.random_seed,
    )

    X_train = X_features.loc[train_pairs_df.index]
    y_train = y_train_full[train_pairs_df.index]

    X_val = X_features.loc[val_pairs_df.index]
    y_val = y_train_full[val_pairs_df.index]
    timer.end_stage("5. Entity-Level Data Split (Zero Leakage)")

    # -------------------------------------------------------------------------
    # 6. MODEL TRAINING
    # -------------------------------------------------------------------------
    timer.start_stage(f"6. Training Hybrid Matcher ({config.classifier_type})")
    matcher = EntityMatcher(
        classifier_type=config.classifier_type,
        random_seed=config.random_seed,
    )
    matcher.fit(X_train, y_train, verbose=config.verbose)

    # Report feature importances
    importances = matcher.get_feature_importances()
    if importances:
        print("\nTop 10 Feature Importances:")
        for i, (fname, fval) in enumerate(list(importances.items())[:10], 1):
            print(f"  {i:2d}. {fname:<30}: {fval:.4f}")

    matcher.save(config.model_save_path)
    timer.end_stage(f"6. Training Hybrid Matcher ({config.classifier_type})")

    # -------------------------------------------------------------------------
    # 7. VALIDATION & THRESHOLD OPTIMIZATION
    # -------------------------------------------------------------------------
    timer.start_stage("7. Validation Evaluation & Macro F0.5 Threshold Search")
    val_probs = matcher.predict_proba(X_val)

    best_threshold, threshold_table = optimize_threshold(
        val_candidate_df=val_pairs_df,
        val_probabilities=val_probs,
        ground_truth_mapping=gt_mapping,
        val_s1_ids=val_s1_ids,
        verbose=config.verbose,
    )
    config.match_threshold = best_threshold

    # Comprehensive evaluation at optimal threshold
    eval_report = comprehensive_evaluation(
        candidate_df=val_pairs_df,
        probabilities=val_probs,
        ground_truth_mapping=gt_mapping,
        all_s1_ids=val_s1_ids,
        s1_metadata_df=s1_clean,
        threshold=best_threshold,
    )

    print("\n" + "=" * 55)
    print("COMPREHENSIVE VALIDATION METRICS (Held-out S1 Set)")
    print("=" * 55)
    macro_m = eval_report["macro_metrics"]
    print(f"Primary Macro F0.5:           {macro_m['macro_f05']:.4f}")
    print(f"Macro Precision:              {macro_m['macro_precision']:.4f}")
    print(f"Macro Recall:                 {macro_m['macro_recall']:.4f}")
    print(f"Total Singleton S1:           {macro_m['total_singletons']}")
    print(f"Correct Singleton Predictions:{macro_m['singletons_correct']}")
    print(f"False Singleton Matches:      {macro_m.get('singletons_false_matches', 0)}")
    print(f"Singleton Accuracy:           {macro_m['singleton_accuracy']:.2%}")
    print(f"Total True Positives (TP):    {macro_m['total_tp']}")
    print(f"Total False Positives (FP):   {macro_m['total_fp']}")
    print(f"Total False Negatives (FN):   {macro_m['total_fn']}")

    pair_m = eval_report["pairwise_metrics"]
    print(f"Pairwise ROC-AUC:             {pair_m['roc_auc']:.4f}")
    print(f"Pairwise Accuracy:            {pair_m['accuracy']:.4f}")
    print(f"Pairwise F1:                  {pair_m['f1']:.4f}")
    print(f"Pairwise F0.5:                {pair_m['f05']:.4f}")

    print("\nCountry Breakdown:")
    for country, c_m in eval_report["country_metrics"].items():
        print(f"  {country:<10} (N={c_m['count']}): Macro F0.5={c_m['macro_f05']:.4f}, Prec={c_m['precision']:.4f}, Rec={c_m['recall']:.4f}")

    print("\nPredicted Match-Count Distribution:")
    for category, count in eval_report["match_distribution"].items():
        print(f"  {category:<18}: {count} ({count/len(val_s1_ids)*100.0:.1f}%)")
    print("=" * 55)

    # Error Analysis
    run_error_analysis(
        candidate_df=val_pairs_df,
        probabilities=val_probs,
        ground_truth_mapping=gt_mapping,
        s1_df=s1_clean,
        target_df=target_clean,
        threshold=best_threshold,
        top_n=3,
        verbose=config.verbose,
    )
    timer.end_stage("7. Validation Evaluation & Macro F0.5 Threshold Search")

    # -------------------------------------------------------------------------
    # 8. TEST SET INFERENCE & SUBMISSION OUTPUT
    # -------------------------------------------------------------------------
    if not run_test_inference:
        print("\nTest inference skipped (run_test_inference=False).")
        return

    timer.start_stage("8. Final Test Inference (S1, S2, S3 with France)")
    logger.info("Loading test dataset from %s...", config.test_dir)
    test_data = loader.load_test_all(nrows=config.max_test_rows)
    test_s1 = test_data["s1"]
    test_s2 = test_data["s2"]
    test_s3 = test_data["s3"]

    ordered_test_s1_ids = test_s1["entity_id"].astype(str).str.strip().tolist()

    logger.info(
        "Loaded Test Records: S1=%d, S2=%d, S3=%d",
        len(test_s1),
        len(test_s2),
        len(test_s3),
    )

    # Normalize test text
    test_s1_clean = preprocessor.process_dataframe(test_s1)
    test_s2_clean = preprocessor.process_dataframe(test_s2)
    test_s3_clean = preprocessor.process_dataframe(test_s3)
    test_target_clean = pd.concat([test_s2_clean, test_s3_clean], ignore_index=True)

    # Fit blocker on test targets
    test_blocker = MultiChannelBlocker(
        max_candidates_per_s1=config.max_candidates_per_s1,
        min_token_len=config.min_token_len,
        filter_by_country=config.filter_by_country,
    )
    test_blocker.fit(test_target_clean)

    # Query test candidates
    test_cand_map, test_hits_map = test_blocker.query_candidates(test_s1_clean)
    test_cand_df, test_b_stats = test_blocker.generate_candidate_dataframe(
        s1_df=test_s1_clean,
        candidate_mapping=test_cand_map,
        hit_counts_mapping=test_hits_map,
    )
    print(test_b_stats.report())

    # Extract test features
    test_X = feature_extractor.extract_features(
        candidate_df=test_cand_df,
        s1_df=test_s1_clean,
        target_df=test_target_clean,
        verbose=config.verbose,
    )

    if config.use_semantic and semantic_model.is_available:
        test_sem = semantic_model.extract_semantic_features(
            candidate_df=test_cand_df,
            s1_df=test_s1_clean,
            target_df=test_target_clean,
            verbose=config.verbose,
        )
        test_X = pd.concat([test_X, test_sem], axis=1)

    # Predict probabilities with trained model
    test_probs = matcher.predict_proba(test_X)

    # Apply validation-optimized threshold to select final matches
    # Allows 0, 1, or multiple matches per S1 entity
    test_predictions = generate_entity_predictions(
        candidate_df=test_cand_df,
        probabilities=test_probs,
        all_s1_ids=set(ordered_test_s1_ids),
        threshold=best_threshold,
    )
    timer.end_stage("8. Final Test Inference (S1, S2, S3 with France)")

    # -------------------------------------------------------------------------
    # 9. OUTPUT GENERATION & VALIDATION
    # -------------------------------------------------------------------------
    timer.start_stage("9. Writing Output Files & Executing Validator")
    out_gen = OutputGenerator(config.output_dir)

    matching_file = out_gen.generate_matching_results(
        predictions_mapping=test_predictions,
        ordered_s1_ids=ordered_test_s1_ids,
        filename="matching_results.tsv",
    )
    candidate_file = out_gen.generate_candidate_pairs(
        candidate_mapping=test_cand_map,
        ordered_s1_ids=ordered_test_s1_ids,
        filename="candidate_pairs.tsv",
    )

    # Determine test source1 path and test_dir for validation
    val_test_s1_path = config.test_s1_path
    val_test_dir = config.test_dir

    if config.max_test_rows is not None:
        sample_test_dir = os.path.join(config.output_dir, "sample_test")
        os.makedirs(sample_test_dir, exist_ok=True)
        sample_s1_path = os.path.join(sample_test_dir, "test_source1.tsv")
        test_s1.to_csv(sample_s1_path, sep="\t", index=False)
        test_s2.to_csv(os.path.join(sample_test_dir, "test_source2.tsv"), sep="\t", index=False)
        test_s3.to_csv(os.path.join(sample_test_dir, "test_source3.tsv"), sep="\t", index=False)
        val_test_s1_path = sample_s1_path
        val_test_dir = sample_test_dir
        logger.info("Sample test directory created for validation: %s", sample_test_dir)

    # Perform internal validation
    passed, errors, warnings = validate_submission_internal(
        matching_path=matching_file,
        candidate_path=candidate_file,
        test_source1_path=val_test_s1_path,
        verbose=config.verbose,
    )

    # Invoke official validator if script exists
    if os.path.isfile(config.validator_script_path):
        call_official_validator(
            validator_script_path=config.validator_script_path,
            matching_path=matching_file,
            candidate_path=candidate_file,
            test_dir=val_test_dir,
            check_ids=False,
        )
    timer.end_stage("9. Writing Output Files & Executing Validator")

    print("\n" + "=" * 70)
    print("FINAL DEVELOPMENT EXPERIMENT")
    print("=" * 70)
    print(f"Development S1 entities:       {len(s1_dev):,}")
    print(f"Training S1 entities:          {len(train_s1_ids):,}")
    print(f"Validation S1 entities:        {len(val_s1_ids):,}\n")
    print(f"Positive links:                {subset_stats.n_positive_links:,}")
    print(f"Singletons:                    {subset_stats.n_singletons:,} ({subset_stats.pct_singletons:.2f}%)\n")
    print("Blocking:")
    print(f"Candidate pairs:               {blocking_stats.total_candidate_pairs:,}")
    print(f"Average candidates/S1:         {blocking_stats.avg_candidates_per_s1:.2f}")
    if blocking_stats.blocking_recall is not None:
        print(f"Blocking recall:               {blocking_stats.blocking_recall * 100:.2f}%")
    print(f"Reduction ratio:               {blocking_stats.reduction_ratio * 100:.4f}%\n")
    print("Model:")
    print(f"{config.classifier_type}\n")
    print(f"Optimal threshold:             {best_threshold:.2f}")
    print(f"Validation Macro F0.5:         {macro_m['macro_f05']:.4f}")
    print(f"Validation Precision:          {macro_m['macro_precision']:.4f}")
    print(f"Validation Recall:             {macro_m['macro_recall']:.4f}")
    print(f"Singleton Accuracy:            {macro_m['singleton_accuracy']:.2%}")
    print("=" * 70)
    print(f"Total Execution Time:          {timer.total_time():.2f}s")
    print(f"Matching Results:              {matching_file}")
    print(f"Candidate Pairs:               {candidate_file}")
    print("=" * 70 + "\n")


def main() -> None:
    """CLI entry point for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Run Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root path containing dataset/train and dataset/test (default: auto-detected)",
    )
    parser.add_argument(
        "--n-source1",
        type=int,
        default=10000,
        help="Number of Source 1 entities for development subset (-1 for full)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="hist_gradient_boosting",
        choices=["hist_gradient_boosting", "random_forest", "logistic_regression"],
        help="Classifier algorithm for hybrid matcher",
    )
    parser.add_argument(
        "--use-semantic",
        action="store_true",
        help="Enable sentence-transformer semantic embedding features",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip final test inference (run validation only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random reproducibility seed",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional row limit when loading train files (for quick sample run)",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional row limit when loading test files (for quick sample run)",
    )
    args = parser.parse_args()

    config_kwargs = {
        "n_source1": args.n_source1,
        "max_train_rows": args.max_train_rows,
        "max_test_rows": args.max_test_rows,
        "classifier_type": args.classifier,
        "use_semantic": args.use_semantic,
        "random_seed": args.seed,
    }
    if args.data_root is not None:
        config_kwargs["data_root"] = args.data_root

    cfg = Config(**config_kwargs)

    run_pipeline(config=cfg, run_test_inference=not args.skip_test)


if __name__ == "__main__":
    main()
