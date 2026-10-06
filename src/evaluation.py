"""Evaluation and threshold optimization module for Business Entity Resolution.

Implements the official macro-averaged entity-level F0.5 metric, singleton-aware scoring,
dynamic threshold optimization, comprehensive diagnostic reporting, and error analysis.
"""

from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score

logger = logging.getLogger(__name__)


def calculate_entity_metrics(
    true_set: Set[str], pred_set: Set[str]
) -> Tuple[float, float, float]:
    """Calculate (precision, recall, F0.5) for a single Source 1 entity.

    Strict Official Rules:
    - true == {} and pred == {} -> F0.5 = 1.0 (correct singleton)
    - true == {} and pred != {} -> F0.5 = 0.0 (false merge on singleton)
    - true != {} and pred == {} -> F0.5 = 0.0 (missed match)
    - true != {} and pred != {}:
        P = TP / |pred|
        R = TP / |true|
        F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    n_true = len(true_set)
    n_pred = len(pred_set)

    if n_true == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0
    if n_true == 0 and n_pred > 0:
        return 0.0, 0.0, 0.0
    if n_true > 0 and n_pred == 0:
        return 0.0, 0.0, 0.0

    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / float(n_pred)
    recall = tp / float(n_true)

    denominator = 0.25 * precision + recall
    f05 = (1.25 * precision * recall) / denominator if denominator > 0.0 else 0.0

    return precision, recall, f05


def generate_entity_predictions(
    candidate_df: pd.DataFrame,
    probabilities: np.ndarray,
    all_s1_ids: Set[str],
    threshold: float,
) -> Dict[str, Set[str]]:
    """Convert candidate pair match probabilities to entity-level prediction sets.

    Allows 0, 1, or multiple matches per S1 entity based on threshold.
    Guarantees every S1 entity exists in returned dictionary.
    """
    predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_s1_ids}

    # Mask candidates meeting threshold
    mask = probabilities >= threshold
    passing_df = candidate_df[mask]

    for s1_id, t_id in zip(
        passing_df["source1_entity_id"], passing_df["target_entity_id"]
    ):
        s1_str = str(s1_id).strip()
        t_str = str(t_id).strip()
        if s1_str in predictions:
            predictions[s1_str].add(t_str)

    return predictions


def macro_f05_evaluation(
    ground_truth_mapping: Dict[str, Set[str]],
    predictions_mapping: Dict[str, Set[str]],
    all_s1_ids: Set[str],
) -> Dict[str, float]:
    """Compute official macro-averaged entity metrics over all Source 1 entities."""
    f05_scores: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    singletons_correct = 0
    total_singletons = 0
    total_non_singletons = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for s1_id in all_s1_ids:
        true_set = ground_truth_mapping.get(s1_id, set())
        pred_set = predictions_mapping.get(s1_id, set())

        p, r, f05 = calculate_entity_metrics(true_set, pred_set)
        precisions.append(p)
        recalls.append(r)
        f05_scores.append(f05)

        if len(true_set) == 0:
            total_singletons += 1
            if len(pred_set) == 0:
                singletons_correct += 1
            else:
                total_fp += len(pred_set)
        else:
            total_non_singletons += 1
            tp = len(true_set & pred_set)
            total_tp += tp
            total_fp += len(pred_set - true_set)
            total_fn += len(true_set - pred_set)

    singleton_acc = (
        (singletons_correct / total_singletons) if total_singletons > 0 else 1.0
    )

    return {
        "macro_f05": float(np.mean(f05_scores)) if f05_scores else 0.0,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "singleton_accuracy": float(singleton_acc),
        "total_singletons": total_singletons,
        "singletons_correct": singletons_correct,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "singletons_false_matches": total_singletons - singletons_correct,
    }


def optimize_threshold(
    val_candidate_df: pd.DataFrame,
    val_probabilities: np.ndarray,
    ground_truth_mapping: Dict[str, Set[str]],
    val_s1_ids: Set[str],
    threshold_grid: Optional[List[float]] = None,
    verbose: bool = True,
) -> Tuple[float, pd.DataFrame]:
    """Search for the threshold that maximizes validation Macro F0.5 across 0.10 to 0.95."""
    if threshold_grid is None:
        threshold_grid = [round(t, 2) for t in np.arange(0.10, 0.96, 0.05)]

    results: List[Dict[str, Any]] = []
    best_threshold = 0.50
    best_f05 = -1.0

    for thresh in threshold_grid:
        preds = generate_entity_predictions(
            val_candidate_df, val_probabilities, val_s1_ids, threshold=thresh
        )
        metrics = macro_f05_evaluation(ground_truth_mapping, preds, val_s1_ids)

        row = {
            "threshold": thresh,
            "macro_f05": metrics["macro_f05"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "singleton_acc": metrics["singleton_accuracy"],
            "total_tp": metrics["total_tp"],
            "total_fp": metrics["total_fp"],
            "total_fn": metrics["total_fn"],
        }
        results.append(row)

        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_threshold = thresh

    perf_df = pd.DataFrame(results)

    if verbose:
        print("\n" + "=" * 82)
        print("VALIDATION THRESHOLD OPTIMIZATION (Primary: Macro F0.5)")
        print("=" * 82)
        header = f"{'Threshold':>10} | {'Macro F0.5':>11} | {'Precision':>10} | {'Recall':>8} | {'Singleton Acc':>14} | {'TP':>6} | {'FP':>6} | {'FN':>6}"
        print(header)
        print("-" * 82)
        for _, r in perf_df.iterrows():
            marker = " <-- BEST" if r["threshold"] == best_threshold else ""
            print(
                f"{r['threshold']:>10.2f} | {r['macro_f05']:>11.4f} | {r['macro_precision']:>10.4f} | "
                f"{r['macro_recall']:>8.4f} | {r['singleton_acc']:>13.2%} | {int(r['total_tp']):>6} | {int(r['total_fp']):>6} | {int(r['total_fn']):>6}{marker}"
            )
        print("=" * 82)
        print(f"Optimal Threshold: {best_threshold:.2f} (Macro F0.5: {best_f05:.4f})\n")

    return best_threshold, perf_df


def comprehensive_evaluation(
    candidate_df: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth_mapping: Dict[str, Set[str]],
    all_s1_ids: Set[str],
    s1_metadata_df: Optional[pd.DataFrame] = None,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Execute complete validation evaluation: macro entity, pairwise, country, distribution."""
    # 1. Macro entity metrics
    preds = generate_entity_predictions(
        candidate_df, probabilities, all_s1_ids, threshold=threshold
    )
    macro_metrics = macro_f05_evaluation(ground_truth_mapping, preds, all_s1_ids)

    # 2. Pairwise metrics
    y_true = candidate_df.get("label", np.zeros(len(candidate_df))).values
    y_pred = (probabilities >= threshold).astype(int)

    roc_auc = (
        roc_auc_score(y_true, probabilities)
        if len(np.unique(y_true)) > 1
        else 0.0
    )
    pair_acc = accuracy_score(y_true, y_pred)
    pair_f1 = f1_score(y_true, y_pred, zero_division=0)

    # Pairwise F0.5
    prec = macro_metrics["total_tp"] / max(1, macro_metrics["total_tp"] + macro_metrics["total_fp"])
    rec = macro_metrics["total_tp"] / max(1, macro_metrics["total_tp"] + macro_metrics["total_fn"])
    pair_f05 = (
        (1.25 * prec * rec) / (0.25 * prec + rec)
        if (0.25 * prec + rec) > 0
        else 0.0
    )

    # 3. Country-wise breakdown
    country_metrics: Dict[str, Dict[str, float]] = {}
    if s1_metadata_df is not None and "country" in s1_metadata_df.columns:
        s1_country_map = dict(
            zip(s1_metadata_df["entity_id"], s1_metadata_df["country"])
        )
        country_groups: Dict[str, Set[str]] = defaultdict(set)
        for s1_id in all_s1_ids:
            c = s1_country_map.get(s1_id, "UNKNOWN")
            country_groups[c].add(s1_id)

        for country, c_s1_ids in country_groups.items():
            c_metrics = macro_f05_evaluation(ground_truth_mapping, preds, c_s1_ids)
            country_metrics[country] = {
                "count": len(c_s1_ids),
                "macro_f05": c_metrics["macro_f05"],
                "precision": c_metrics["macro_precision"],
                "recall": c_metrics["macro_recall"],
            }

    # 4. Match-count distribution
    match_distribution = {"0 (singletons)": 0, "1": 0, "2": 0, "3+": 0}
    for s1_id, matched in preds.items():
        nm = len(matched)
        if nm == 0:
            match_distribution["0 (singletons)"] += 1
        elif nm == 1:
            match_distribution["1"] += 1
        elif nm == 2:
            match_distribution["2"] += 1
        else:
            match_distribution["3+"] += 1

    report = {
        "macro_metrics": macro_metrics,
        "pairwise_metrics": {
            "roc_auc": roc_auc,
            "accuracy": pair_acc,
            "f1": pair_f1,
            "f05": pair_f05,
        },
        "country_metrics": country_metrics,
        "match_distribution": match_distribution,
    }
    return report


def run_error_analysis(
    candidate_df: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth_mapping: Dict[str, Set[str]],
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    threshold: float,
    top_n: int = 5,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Inspect top false positive and false negative pairs for model diagnostics."""
    df = candidate_df.copy()
    df["probability"] = probabilities
    df["predicted_match"] = (df["probability"] >= threshold).astype(int)

    # Index text
    s1_text = s1_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    tgt_text = target_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")

    # False Positives: predicted 1 but true 0
    fp_df = df[(df["predicted_match"] == 1) & (df["label"] == 0)].sort_values(
        by="probability", ascending=False
    ).head(top_n)

    # False Negatives: true 1 but predicted 0
    fn_df = df[(df["predicted_match"] == 0) & (df["label"] == 1)].sort_values(
        by="probability", ascending=True
    ).head(top_n)

    def enrich(err_df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for _, row in err_df.iterrows():
            s1 = s1_text.get(row["source1_entity_id"], {})
            tgt = tgt_text.get(row["target_entity_id"], {})
            out.append({
                "source1_id": row["source1_entity_id"],
                "s1_name": s1.get("business_name", ""),
                "s1_address": s1.get("business_address", ""),
                "target_id": row["target_entity_id"],
                "target_name": tgt.get("business_name", ""),
                "target_address": tgt.get("business_address", ""),
                "probability": row["probability"],
                "label": row["label"],
            })
        return pd.DataFrame(out)

    fp_enriched = enrich(fp_df)
    fn_enriched = enrich(fn_df)

    safe_str = lambda s: str(s).encode("ascii", errors="replace").decode("ascii")

    if verbose:
        print("\n" + "=" * 70)
        print("ERROR ANALYSIS: TOP FALSE POSITIVES (Highest Confidence Non-Matches)")
        print("=" * 70)
        for _, r in fp_enriched.iterrows():
            print(f"Prob: {r['probability']:.4f} | S1: [{r['source1_id']}] {safe_str(r['s1_name'])} -- {safe_str(r['s1_address'])}")
            print(f"               | TG: [{r['target_id']}] {safe_str(r['target_name'])} -- {safe_str(r['target_address'])}\n")

        print("=" * 70)
        print("ERROR ANALYSIS: TOP FALSE NEGATIVES (Lowest Confidence True Matches)")
        print("=" * 70)
        for _, r in fn_enriched.iterrows():
            print(f"Prob: {r['probability']:.4f} | S1: [{r['source1_id']}] {safe_str(r['s1_name'])} -- {safe_str(r['s1_address'])}")
            print(f"               | TG: [{r['target_id']}] {safe_str(r['target_name'])} -- {safe_str(r['target_address'])}\n")
        print("=" * 70)

    return fp_enriched, fn_enriched
