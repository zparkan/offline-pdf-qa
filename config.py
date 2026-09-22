import os
import json
import shutil
from pathlib import Path
from model_manager import ModelManager

# =====================================================================
# ۱. تنظیمات محیط و مسیرها (Paths & Environment)
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_ROOT = BASE_DIR / "models"
DB_DIR = BASE_DIR / "chroma_db"
DB_INFO_FILE = DB_DIR / "db_info.json"

# پوشه داده‌ها (برای چانک‌ها، PDFها و ذخیره‌سازی داده‌های استخراج شده فاطمه)
DATA_DIR = BASE_DIR / "data"

# مسیر پایگاه داده چت‌ها (نام موقت: chat_history.db - بعد از پیاده‌سازی نهایی بررسی شود)
CHAT_DB_FILE = BASE_DIR / "chat_history.db"

# تشخیص محیط اجرا (Google Colab یا سیستم محلی)
IS_COLAB = "google.colab" in str(os.environ)
ENV_MODE = "COLAB" if IS_COLAB else "LOCAL"

# =====================================================================
# ۲. ریجستری مدل‌های امبدینگ (Top Models & cheRAGh Ranking)
# =====================================================================
EMBEDDING_MODELS = {
    # مدل سبک پیش‌فرض برای توسعه و سیستم محلی
    "e5-small": {
        "repo_id": "intfloat/multilingual-e5-small",
        "dim": 384,
        "desc": "مدل بسیار سبک و سریع E5 - عالی برای تست و سیستم محلی"
    },
    # سایر مدل‌های برتر cheRAGh و HuggingFace
    "qwen3-emb-4b": {
        "repo_id": "Qwen/Qwen3-Embedding-4B",
        "dim": 2560,
        "desc": "مدل بسیار قوی و مدرن سری کوئن ۳ برای امبدینگ"
    },
    "bge-m3": {
        "repo_id": "BAAI/bge-m3",
        "dim": 1024,
        "desc": "رتبه ۲ cheRAGh - چندزبانه همه‌فن‌حریف با درک عمیق فارسی"
    },
    "octen-emb-8b": {
        "repo_id": "Octen/Octen-Embedding-8B",
        "dim": 4096,
        "desc": "مدل بسیار بزرگ و قدرتمند ۸ میلیاردی برای ارزیابی نهایی"
    },
    "qwen3-emb-8b": {
        "repo_id": "Qwen/Qwen3-Embedding-8B",
        "dim": 4096,
        "desc": "نسخه ۸ میلیارد پارامتری کوئن ۳ برای بیشترین دقت معنایی"
    },
    "snowflake-arctic-l": {
        "repo_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "dim": 1024,
        "desc": "مدل قدرتمند اسنوفلیک نسخه ۲ برای جستجوی دقیق معنایی"
    },
    "f2llm-4b": {
        "repo_id": "codefuse-ai/F2LLM-v2-4B",
        "dim": 2560,
        "desc": "مدل تخصصی بر پایه LLM برای بازنمایی برداری"
    },
    "e5-large": {
        "repo_id": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "desc": "رتبه ۱ cheRAGh - بسیار دقیق در متون چندزبانه"
    },
    "e5-base": {
        "repo_id": "intfloat/multilingual-e5-base",
        "dim": 768,
        "desc": "تعادل عالی بین سرعت، حافظه و دقت معنایی"
    },
    "fa-sentence": {
        "repo_id": "MirSamanPR/fa-sentence-v2",
        "dim": 768,
        "desc": "رتبه ۳ cheRAGh - مدل آموزش‌دیده اختصاصی برای زبان فارسی"
    },
    "minilm": {
        "repo_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "desc": "بسیار سبک و سریع، مناسب سیستم‌های ضعیف با رم پایین"
    }
}

# =====================================================================
# ۳. ریجستری مدل‌های زبانی (فقط مبتنی بر llama-cpp و فایل‌های GGUF)
# =====================================================================
LLM_MODELS = {
    "qwen-3b": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "desc": "کوئن ۲.۵ سه میلیارد پارامتری سبک، سریع برای اجرای یکپارچه"
    },
    "qwen-7b": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "desc": "کیفیت بالا برای زبان فارسی"
    },
    "qwen-3.5-8b": {
        "repo": "Qwen/Qwen3.5-8B-GGUF",
        "filename": "qwen3.5-8b-q4_k_m.gguf",
        "desc": "نسخه جدید کوئن برای بنچمارک"
    },
    "deepseek-14b": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "filename": "deepseek-r1-distill-qwen-14b-q4_k_m.gguf",
        "desc": "استدلال عمیق، مناسب کولب با گرافیک بالا"
    }
}

# =====================================================================
# ۴. انتخاب مدل‌های فعال (Active Selection)
# =====================================================================
ACTIVE_EMBEDDING = "e5-small"  # مدل امبدینگ فعال: e5-small برای سبکی و سرعت
ACTIVE_LLM = "qwen-3b"        # مدل زبانی فعال یکپارچه با llama-cpp

# =====================================================================
# ۵. توابع مدیریت دیتابیس و مدل‌ها (Database & Model Helpers)
# =====================================================================

def save_db_info(model_name: str):
    """
    توضیح:
        ذخیره نام مدل امبدینگ فعلی داخل فایل db_info.json کنار پایگاه داده برداری.
    ورودی:
        model_name (str): کلید مدل امبدینگ (مثلاً 'e5-small')
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    with open(DB_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump({"embedding_model": model_name}, f, ensure_ascii=False, indent=2)

def get_current_db_model() -> str:
    """
    توضیح:
        خواندن نام مدلی که دیتابیس برداری فعلی با آن ساخته شده است.
    خروجی:
        (str یا None): نام مدل ذخیره شده در db_info.json، یا None در صورت نبود دیتابیس.
    """
    if DB_INFO_FILE.exists():
        try:
            with open(DB_INFO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("embedding_model")
        except Exception:
            return None
    return None

def clear_all_databases():
    """
    توضیح:
        ۱) کل پوشه پایگاه داده برداری (ChromaDB) را پاک می‌کند.
        ۲) محتویات داخل پایگاه داده چت‌ها را پاک می‌کند (نه ساختار یا جدول‌ها را).
    """
    # الف) پاکسازی پایگاه داده برداری
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        print("[!] پایگاه داده برداری (ChromaDB) پاکسازی شد.")

    # ب) پاکسازی اطلاعات داخل پایگاه چت‌ها
    try:
        from chat_database import clear_chat_records
        clear_chat_records()
        print("[!] اطلاعات تاریخچه چت‌ها پاکسازی شد.")
    except Exception:
        if CHAT_DB_FILE.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(CHAT_DB_FILE)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                for table_name in tables:
                    if not table_name[0].startswith("sqlite_"):
                        cursor.execute(f"DELETE FROM {table_name[0]};")
                conn.commit()
                conn.close()
                print("[!] داده‌های جداول دیتابیس چت پاک شد.")
            except Exception as err:
                print(f"[!] خطا در پاکسازی اطلاعات پایگاه داده چت: {err}")

def set_active_embedding(new_model_key: str) -> bool:
    """
    توضیح:
        تنظیم مدل امبدینگ فعال.
        اگر با مدل قبلیِ دیتابیس برداری تفاوت داشته باشد:
        - پایگاه داده برداری را کاملاً پاک می‌کند.
        - رکوردهای پایگاه داده چت را پاک می‌کند.
        - فایل اطلاعات دیتابیس (db_info.json) را با مدل جدید به‌روزرسانی می‌کند.
    ورودی:
        new_model_key (str): نام کلید مدل از لیست EMBEDDING_MODELS (مثلاً 'e5-small')
    """
    global ACTIVE_EMBEDDING

    if new_model_key not in EMBEDDING_MODELS:
        print(f"[!] مدل '{new_model_key}' در لیست مدل‌های امبدینگ یافت نشد.")
        return False

    ACTIVE_EMBEDDING = new_model_key
    print(f"[*] مدل امبدینگ فعال روی '{new_model_key}' تنظیم شد.")

    db_model = get_current_db_model()

    if db_model is not None and db_model != new_model_key:
        print(f"[!] تغییر مدل دیتابیس تشخیص داده شد ({db_model} -> {new_model_key}).")
        clear_all_databases()
        save_db_info(new_model_key)
    elif db_model is None:
        save_db_info(new_model_key)

    return True

def setup_infrastructure():
    """
    توضیح:
        بررسی آماده بودن فایل‌های مدل امبدینگ و مدل زبانی (GGUF محلی).
    """
    manager = ModelManager(root_dir=MODELS_ROOT)

    # بررسی مدل امبدینگ فعال
    emb_info = EMBEDDING_MODELS[ACTIVE_EMBEDDING]
    success, msg, _ = manager.get_embedding_model(emb_info["repo_id"], ACTIVE_EMBEDDING)
    print(f"[*] وضعیت مدل امبدینگ ({ACTIVE_EMBEDDING}): {msg}")

    # بررسی مدل زبانی فعال به صورت فایل محلی GGUF یکپارچه
    llm_info = LLM_MODELS[ACTIVE_LLM]
    success, msg, _ = manager.setup_llm(
        mode="local",
        model_name_or_repo=llm_info["repo"],
        filename=llm_info.get("filename")
    )
    print(f"[*] وضعیت مدل زبانی ({ACTIVE_LLM}): {msg}")

if __name__ == "__main__":
    setup_infrastructure()
