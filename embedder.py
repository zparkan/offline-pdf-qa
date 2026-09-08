# embedder.py
from __future__ import annotations

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

import config  # EMBEDDING_MODEL_NAME, EMBEDDING_DIM, MODELS_DIR


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
        embedder.unload_model()  # before loading LLM
    """

    _instance: Embedder | None = None

    def __init__(self) -> None:
        model_path = Path(config.MODELS_DIR) / config.EMBEDDING_MODEL_KEY
        load_target = (
            str(model_path) if model_path.exists() else config.EMBEDDING_MODEL_NAME
        )

        print(f"[Embedder] Loading model from: {load_target}")
        self._model: SentenceTransformer | None = SentenceTransformer(load_target)
        print(f"[Embedder] Ready. Vector dim = {config.EMBEDDING_DIM}")

    @classmethod
    def get_instance(cls) -> "Embedder":
        """Return the shared Embedder, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_passages(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of document chunks for storage in ChromaDB.

        The mandatory "passage:" prefix tells the e5 model that these are
        documents to be indexed, not queries.

        Returns:
            np.ndarray of shape (len(texts), EMBEDDING_DIM), L2-normalised.
        """
        self._require_model()
        model = self._model
        if model is None:
            raise RuntimeError("Model not loaded.")  # دفاعی، _require_model قبلاً چک کرده
        prefixed = [f"passage: {t}" for t in texts]
        return model.encode(  # ← از متغیر محلی استفاده می‌کنیم، نه self._model
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single user query for similarity search.

        The "query:" prefix puts the e5 model in retrieval mode.

        Returns:
            np.ndarray of shape (EMBEDDING_DIM,), L2-normalised.
        """
        self._require_model()
        model = self._model
        if model is None:
            raise RuntimeError("Model not loaded.")  # ← indentation درست شد
        prefixed = f"query: {text}"
        return model.encode(  # ← از متغیر محلی استفاده می‌کنیم
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def unload_model(self) -> None:
        """
        Delete the model from RAM (and VRAM if using GPU).

        Call this right before loading the LLM so both models don't compete
        for memory.
        """
        if self._model is not None:
            del self._model
            self._model = None
            Embedder._instance = None
            print("[Embedder] Model unloaded.")
        else:
            print("[Embedder] Nothing to unload.")

    def _require_model(self) -> None:
        """Raise a clear error if someone calls embed_* after unload_model()."""
        if self._model is None:
            raise RuntimeError(
                "[Embedder] Model has been unloaded. "
                "Call Embedder.get_instance() again to reload."
            )
