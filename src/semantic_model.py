"""Semantic embedding module for Business Entity Resolution.

Provides an optional, license-compliant sentence-transformer feature extractor
(Apache 2.0, ~22M params) with deduplicated caching and automatic GPU detection.
"""

import logging
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HAS_SEMANTIC_DEPS: Optional[bool] = None


class SemanticModel:
    """Computes dense cosine similarity features using Apache 2.0 licensed SentenceTransformer."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 256,
        max_sequence_length: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Lazily initialize the sentence-transformer encoder."""
        global HAS_SEMANTIC_DEPS
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            HAS_SEMANTIC_DEPS = True
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Initializing SemanticModel [%s] on device: %s", self.model_name, self.device)
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model.max_seq_length = self.max_sequence_length
        except ImportError:
            HAS_SEMANTIC_DEPS = False
            logger.info("sentence-transformers or torch not installed. Semantic modeling disabled.")
            self.model = None
        except Exception as exc:
            HAS_SEMANTIC_DEPS = False
            logger.warning("Could not initialize SentenceTransformer (%s). Falling back.", exc)
            self.model = None

    @property
    def is_available(self) -> bool:
        """Check if semantic encoder is initialized and operational."""
        return self.model is not None

    def encode_unique_texts(
        self, texts: Set[str], verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """Encode unique strings in batches, caching unit-normalized embeddings."""
        if not self.is_available:
            return {}

        to_encode = [t for t in texts if t and t not in self.embedding_cache]
        if not to_encode:
            return self.embedding_cache

        if verbose:
            logger.info(
                "Encoding %d unique strings with %s (batch_size=%d)...",
                len(to_encode),
                self.model_name,
                self.batch_size,
            )

        # Generate L2-normalized embeddings
        embeddings = self.model.encode(
            to_encode,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        for text, emb in zip(to_encode, embeddings):
            self.embedding_cache[text] = emb

        return self.embedding_cache

    def extract_semantic_features(
        self,
        candidate_df: pd.DataFrame,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Compute cosine similarity for business names and addresses."""
        if not self.is_available:
            # Return neutral 0.0 features if semantic model is unavailable
            return pd.DataFrame(
                {
                    "semantic_name_cosine": np.zeros(len(candidate_df), dtype=np.float32),
                    "semantic_addr_cosine": np.zeros(len(candidate_df), dtype=np.float32),
                },
                index=candidate_df.index,
            )

        # Index text fields by ID
        s1_dict = s1_df.set_index("entity_id")[["normalized_name", "normalized_address"]].to_dict(orient="index")
        target_dict = target_df.set_index("entity_id")[["normalized_name", "normalized_address"]].to_dict(orient="index")

        # Collect unique strings required for this batch of candidate pairs
        unique_names: Set[str] = set()
        unique_addrs: Set[str] = set()

        s1_ids = candidate_df["source1_entity_id"].tolist()
        t_ids = candidate_df["target_entity_id"].tolist()

        for s1_id, t_id in zip(s1_ids, t_ids):
            r1 = s1_dict.get(s1_id, {})
            r2 = target_dict.get(t_id, {})
            n1 = r1.get("normalized_name", "")
            n2 = r2.get("normalized_name", "")
            a1 = r1.get("normalized_address", "")
            a2 = r2.get("normalized_address", "")

            if n1:
                unique_names.add(n1)
            if n2:
                unique_names.add(n2)
            if a1:
                unique_addrs.add(a1)
            if a2:
                unique_addrs.add(a2)

        # Encode unique names and addresses once
        self.encode_unique_texts(unique_names, verbose=verbose)
        self.encode_unique_texts(unique_addrs, verbose=verbose)

        # Compute dot product between normalized embeddings (cosine similarity)
        name_sims: List[float] = []
        addr_sims: List[float] = []

        for s1_id, t_id in zip(s1_ids, t_ids):
            r1 = s1_dict.get(s1_id, {})
            r2 = target_dict.get(t_id, {})

            n1 = r1.get("normalized_name", "")
            n2 = r2.get("normalized_name", "")
            a1 = r1.get("normalized_address", "")
            a2 = r2.get("normalized_address", "")

            # Name similarity
            if n1 and n2 and n1 in self.embedding_cache and n2 in self.embedding_cache:
                sim_n = float(np.dot(self.embedding_cache[n1], self.embedding_cache[n2]))
                name_sims.append(max(0.0, min(1.0, sim_n)))
            else:
                name_sims.append(0.0)

            # Address similarity
            if a1 and a2 and a1 in self.embedding_cache and a2 in self.embedding_cache:
                sim_a = float(np.dot(self.embedding_cache[a1], self.embedding_cache[a2]))
                addr_sims.append(max(0.0, min(1.0, sim_a)))
            else:
                addr_sims.append(0.0)

        sem_df = pd.DataFrame(
            {
                "semantic_name_cosine": name_sims,
                "semantic_addr_cosine": addr_sims,
            },
            index=candidate_df.index,
        )
        return sem_df
