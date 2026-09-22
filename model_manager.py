import os
import subprocess
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

class ModelManager:
    def __init__(self, root_dir="models"):
        self.root_dir = Path(root_dir)
        self.embedding_dir = self.root_dir / "embedding"
        self.llm_dir = self.root_dir / "llm"
        self._create_directories()

    def _create_directories(self):
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.llm_dir.mkdir(parents=True, exist_ok=True)

    def get_embedding_model(self, repo_id, model_name):
        target_path = self.embedding_dir / model_name
        if target_path.exists() and any(target_path.iterdir()):
            return True, f"EXIST: {model_name}", str(target_path)
        
        try:
            model = SentenceTransformer(repo_id)
            model.save(str(target_path))
            return True, f"DOWNLOADED: {model_name}", str(target_path)
        except Exception as e:
            return False, str(e), None

    def setup_llm(self, mode, model_name_or_repo, filename=None):
        if mode == "ollama":
            try:
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
                if model_name_or_repo in result.stdout:
                    return True, f"OLLAMA READY: {model_name_or_repo}", "ollama"
                else:
                    return False, f"OLLAMA MODEL NOT FOUND: {model_name_or_repo}", None
            except Exception:
                return False, "OLLAMA NOT INSTALLED", None

        elif mode == "local":
            target_file = self.llm_dir / filename
            if target_file.exists():
                return True, f"LOCAL READY: {filename}", str(target_file)
            return False, f"LOCAL FILE NOT FOUND: {filename}", None
