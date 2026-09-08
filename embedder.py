# embedder.py
"""
Embedder module for offline RAG system.
Handles text embedding using multilingual-e5 models with mandatory prefixes.

Design:
- Singleton pattern: model loads once, reused across calls
- Manual embedding: bypasses ChromaDB auto-embedding (required for e5 prefixes)
- L2 normalization: all output vectors have unit length (cosine similarity = dot product)
- Batch encoding: handles long document lists efficiently
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

import config  # provides: EMBEDDING_MODEL, VECTOR_DIM, MODELS_DIR


class Embedder:
    """
    Singleton wrapper around a SentenceTransformer model.

    Why Singleton?
    Loading a ~120MB (small) or ~1.1GB (large) model takes several seconds
    and eats RAM. We load once and reuse the same object everywhere.

    Usage:
        embedder = Embedder.get_instance()
        vecs = embedder.embed_passages(["متن اول", "متن دوم"])
        q_vec = embedder.embed_query("سوال کاربر")
        embedder.unload_model()   # before loading LLM
    """

    _instance: Embedder | None = None   # the single shared object

    # ------------------------------------------------------------------ #
    #  Singleton factory                                                   #
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        # Called only once thanks to get_instance()
        model_path = Path(config.MODELS_DIR) / config.EMBEDDING_MODEL

        # Fall back to HuggingFace Hub if the local copy doesn't exist yet.
        # During normal operation save_models.py will have already cached it.
        load_target = str(model_path) if model_path.exists() else config.EMBEDDING_MODEL

        print(f"[Embedder] Loading model from: {load_target}")
        self._model: SentenceTransformer | None = SentenceTransformer(load_target)
        print(f"[Embedder] Ready. Vector dim = {config.VECTOR_DIM}")

    @classmethod
    def get_instance(cls) -> "Embedder":
        """Return the shared Embedder, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def embed_passages(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of document chunks for storage in ChromaDB.

        The mandatory "passage:" prefix tells the e5 model that these are
        *documents to be indexed*, not queries. Without it accuracy drops
        significantly.

        Args:
            texts:         List of raw chunk texts (no prefix needed from caller).
            batch_size:    How many texts to encode at once. Reduce if you hit
                           an out-of-memory error on large documents.
            show_progress: Print a tqdm bar (useful for large collections).

        Returns:
            np.ndarray of shape (len(texts), VECTOR_DIM), L2-normalised.
        """
        self._require_model()
        prefixed = [f"passage: {t}" for t in texts]
        vectors = self._model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 norm built-in
        )
        return vectors.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single user query for similarity search.

        The "query:" prefix puts the e5 model in *retrieval* mode, which is
        different from the passage mode above. Mixing prefixes degrades recall.

        Args:
            text: The raw user question (no prefix needed from caller).

        Returns:
            np.ndarray of shape (VECTOR_DIM,), L2-normalised.
        """
        self._require_model()
        prefixed = f"query: {text}"
        vector = self._model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.astype(np.float32)

    def unload_model(self) -> None:
        """
        Delete the model from RAM (and VRAM if using GPU).

        Call this right before loading the LLM so both models don't compete
        for memory. After calling this, get_instance() will reload the model
        on the next call — that's intentional.
        """
        if self._model is not None:
            del self._model
            self._model = None
            Embedder._instance = None   # allow re-instantiation after reload
            print("[Embedder] Model unloaded.")
        else:
            print("[Embedder] Nothing to unload.")

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _require_model(self) -> None:
        """Raise a clear error if someone calls embed_* after unload_model()."""
        if self._model is None:
            raise RuntimeError(
                "[Embedder] Model has been unloaded. "
                "Call Embedder.get_instance() again to reload."
            )
