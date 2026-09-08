# config.py
import os

# ─── تنظیمات مدل امبدینگ ───────────────────────────────────────────
# برای تغییر مدل، متغیر محیطی EMBEDDING_MODEL رو ست کن
# یا مستقیم مقدار DEFAULT_MODEL رو عوض کن

AVAILABLE_MODELS = {
    "small": {
        "name": "intfloat/multilingual-e5-small",
        "dim": 384,
        "description": "سبک، مناسب برای توسعه (۸GB RAM)"
    },
    "base": {
        "name": "intfloat/multilingual-e5-base",
        "dim": 768,
        "description": "متوسط، تعادل سرعت و دقت"
    },
    "large": {
        "name": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "description": "دقیق‌تر، مناسب برای تست نهایی (۱۶GB RAM)"
    },
}

# اگر EMBEDDING_MODEL ست نشده باشه، از small استفاده می‌کنه
_model_key = os.environ.get("EMBEDDING_MODEL", "small")

if _model_key not in AVAILABLE_MODELS:
    raise ValueError(
        f"مدل '{_model_key}' معتبر نیست. "
        f"گزینه‌های موجود: {list(AVAILABLE_MODELS.keys())}"
    )

EMBEDDING_MODEL_NAME: str = AVAILABLE_MODELS[_model_key]["name"]
EMBEDDING_DIM: int = AVAILABLE_MODELS[_model_key]["dim"]

# ─── مسیرها ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
CHROMA_DIR   = os.path.join(PROJECT_ROOT, "chroma_db")

# ─── تنظیمات ChromaDB ──────────────────────────────────────────────
CHROMA_COLLECTION_PREFIX = "chat_"   # نام کالکشن = chat_{chat_id}

# ─── تنظیمات LLM ───────────────────────────────────────────────────
LLM_MODEL_NAME = "qwen3"
LLM_PARAMS     = "7B"
LLM_QUANTIZE   = "Q4"
