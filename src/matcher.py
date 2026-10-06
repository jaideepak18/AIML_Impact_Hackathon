"""Hybrid matching model module for Business Entity Resolution.

Implements supervised classifier training on tabular lexical, semantic,
and structural features with entity-level data splitting (zero leakage),
calibrated probability estimation, and feature importance reporting.
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def entity_level_train_val_split(
    all_s1_ids: Any,
    candidate_df: Optional[pd.DataFrame] = None,
    validation_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str], Set[str]]:
    """Partition candidate pairs at the Source 1 entity level to prevent data leakage.

    Accepts the complete set of development Source 1 entity IDs so that singletons
    or entities with 0 candidates are NEVER silently dropped during evaluation.

    Asserts: train_count + validation_count == total_count.
    """
    if candidate_df is None and isinstance(all_s1_ids, pd.DataFrame):
        candidate_df = all_s1_ids
        s1_iterable = candidate_df["source1_entity_id"].unique()
    elif isinstance(all_s1_ids, pd.DataFrame):
        s1_iterable = all_s1_ids["entity_id"].unique() if "entity_id" in all_s1_ids.columns else all_s1_ids.iloc[:, 0].unique()
    elif isinstance(all_s1_ids, (pd.Series, list, set, tuple, np.ndarray)):
        s1_iterable = all_s1_ids
    else:
        raise ValueError(f"Unsupported type for all_s1_ids: {type(all_s1_ids)}")

    unique_s1 = sorted(list({str(x).strip() for x in s1_iterable if str(x).strip()}))
    total_count = len(unique_s1)

    rng = np.random.RandomState(random_seed)
    shuffled_s1 = rng.permutation(unique_s1)

    n_val = int(total_count * validation_size)
    val_s1_ids = set(shuffled_s1[:n_val])
    train_s1_ids = set(shuffled_s1[n_val:])

    train_count = len(train_s1_ids)
    val_count = len(val_s1_ids)

    # Strictly assert no entity is lost
    assert train_count + val_count == total_count, (
        f"Train/Val split entity count mismatch: {train_count} + {val_count} != {total_count}"
    )

    print("-" * 55)
    print("ENTITY-LEVEL DATA SPLIT (Zero Leakage)")
    print("-" * 55)
    print(f"Total S1:            {total_count:,}")
    print(f"Train S1:            {train_count:,} ({train_count / total_count * 100:.1f}%)")
    print(f"Validation S1:       {val_count:,} ({val_count / total_count * 100:.1f}%)")
    print(f"Train + Validation:  {train_count + val_count:,}")
    print("-" * 55)

    if candidate_df is not None:
        train_mask = candidate_df["source1_entity_id"].isin(train_s1_ids)
        val_mask = candidate_df["source1_entity_id"].isin(val_s1_ids)
        train_df = candidate_df[train_mask].copy()
        val_df = candidate_df[val_mask].copy()
    else:
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()

    logger.info(
        "Entity-level Split: %d Train S1 (%d pairs) | %d Val S1 (%d pairs)",
        train_count,
        len(train_df),
        val_count,
        len(val_df),
    )

    return train_df, val_df, train_s1_ids, val_s1_ids


class EntityMatcher:
    """Supervised hybrid entity resolution matching model."""

    def __init__(
        self,
        classifier_type: str = "hist_gradient_boosting",
        random_seed: int = 42,
    ) -> None:
        self.classifier_type = classifier_type.lower()
        self.random_seed = random_seed
        self.feature_names: List[str] = []
        self.model = self._create_model()

    def _create_model(self):
        """Instantiate the underlying scikit-learn classifier."""
        if self.classifier_type == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=self.random_seed,
            )
        elif self.classifier_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.random_seed,
            )
        elif self.classifier_type == "logistic_regression":
            return Pipeline([
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        C=1.0,
                        max_iter=1000,
                        random_state=self.random_seed,
                    ),
                ),
            ])
        else:
            raise ValueError(
                f"Unsupported classifier_type '{self.classifier_type}'. "
                f"Choose from: 'hist_gradient_boosting', 'random_forest', 'logistic_regression'"
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        verbose: bool = True,
    ) -> "EntityMatcher":
        """Train the classifier on the candidate pair feature matrix."""
        self.feature_names = list(X.columns)
        if verbose:
            n_pos = int(np.sum(y == 1))
            n_neg = int(np.sum(y == 0))
            logger.info(
                "Training %s on %d samples (Pos: %d, Neg: %d, Feats: %d)...",
                self.classifier_type,
                len(X),
                n_pos,
                n_neg,
                len(self.feature_names),
            )

        self.model.fit(X.values, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate match probabilities P(match=1 | features)."""
        # Ensure column ordering matches training
        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]

        probabilities = self.model.predict_proba(X.values)[:, 1]
        return np.clip(probabilities, 0.0, 1.0)

    def get_feature_importances(self) -> Dict[str, float]:
        """Retrieve relative feature importances if available on model."""
        importances: Dict[str, float] = {}

        if hasattr(self.model, "feature_importances_"):
            vals = self.model.feature_importances_
            for name, val in zip(self.feature_names, vals):
                importances[name] = float(val)
        elif self.classifier_type == "logistic_regression":
            coefs = np.abs(self.model.named_steps["clf"].coef_[0])
            for name, val in zip(self.feature_names, coefs):
                importances[name] = float(val)

        # Sort descending
        return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    def save(self, filepath: str) -> None:
        """Serialize trained model to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "classifier_type": self.classifier_type,
                "feature_names": self.feature_names,
                "random_seed": self.random_seed,
            }, f)
        logger.info("Saved trained EntityMatcher to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "EntityMatcher":
        """Deserialize trained model from disk."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            classifier_type=data["classifier_type"],
            random_seed=data["random_seed"],
        )
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        logger.info("Loaded EntityMatcher from %s", filepath)
        return instance
