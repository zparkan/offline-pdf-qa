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
EMBEDDING_DIR = MODELS_ROOT / "embedding"
LLM_DIR = MODELS_ROOT / "llm"
DB_DIR = BASE_DIR / "chroma_db"
DB_INFO_FILE = DB_DIR / "db_info.json"

# پوشه داده‌ها (برای چانک‌ها، PDFها و ذخیره‌سازی داده‌های استخراج شده فاطمه)
DATA_DIR = BASE_DIR / "data"

# مسیر پایگاه داده چت‌ها (هماهنگ با chat_database.py در پوشه database)
DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
CHAT_DB_FILE = DATABASE_DIR / "chat_history.db"

# پورت سرور وب رابط کاربری (پورت 8050 انتخاب شد تا با Connect.exe و پورت 8080 تداخل نداشته باشد)
SERVER_PORT = 8050

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
        "max_tokens": 512,
        "max_chars": 1200,
        "desc": "مدل بسیار سبک و سریع E5 - عالی برای تست و سیستم محلی"
    },
    # سایر مدل‌های برتر cheRAGh و HuggingFace
    "qwen3-emb-4b": {
        "repo_id": "Qwen/Qwen3-Embedding-4B",
        "dim": 2560,
        "max_tokens": 32768,
        "max_chars": 75000,
        "desc": "مدل بسیار قوی و مدرن سری کوئن ۳ برای امبدینگ"
    },
    "bge-m3": {
        "repo_id": "BAAI/bge-m3",
        "dim": 1024,
        "max_tokens": 8192,
        "max_chars": 18000,
        "desc": "رتبه ۲ cheRAGh - چندزبانه همه‌فن‌حریف با درک عمیق فارسی"
    },
    "octen-emb-8b": {
        "repo_id": "Octen/Octen-Embedding-8B",
        "dim": 4096,
        "max_tokens": 8192,
        "max_chars": 18000,
        "desc": "مدل بسیار بزرگ و قدرتمند ۸ میلیاردی برای ارزیابی نهایی"
    },
    "qwen3-emb-8b": {
        "repo_id": "Qwen/Qwen3-Embedding-8B",
        "dim": 4096,
        "max_tokens": 32768,
        "max_chars": 75000,
        "desc": "نسخه ۸ میلیارد پارامتری کوئن ۳ برای بیشترین دقت معنایی"
    },
    "snowflake-arctic-l": {
        "repo_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "dim": 1024,
        "max_tokens": 8192,
        "max_chars": 18000,
        "desc": "مدل قدرتمند اسنوفلیک نسخه ۲ برای جستجوی دقیق معنایی"
    },
    "f2llm-4b": {
        "repo_id": "codefuse-ai/F2LLM-v2-4B",
        "dim": 2560,
        "max_tokens": 4096,
        "max_chars": 9000,
        "desc": "مدل تخصصی بر پایه LLM برای بازنمایی برداری"
    },
    "e5-large": {
        "repo_id": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "max_tokens": 512,
        "max_chars": 1200,
        "desc": "رتبه ۱ cheRAGh - بسیار دقیق در متون چندزبانه"
    },
    "e5-base": {
        "repo_id": "intfloat/multilingual-e5-base",
        "dim": 768,
        "max_tokens": 512,
        "max_chars": 1200,
        "desc": "تعادل عالی بین سرعت، حافظه و دقت معنایی"
    },
    "fa-sentence": {
        "repo_id": "MirSamanPR/fa-sentence-v2",
        "dim": 768,
        "max_tokens": 512,
        "max_chars": 1200,
        "desc": "رتبه ۳ cheRAGh - مدل آموزش‌دیده اختصاصی برای زبان فارسی"
    },
    "minilm": {
        "repo_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "max_tokens": 128,
        "max_chars": 300,
        "desc": "بسیار سبک و سریع، مناسب سیستم‌های ضعیف با رم پایین"
    }
}

# =====================================================================
# ۳. ریجستری مدل‌های زبانی (فقط مبتنی بر llama-cpp و فایل‌های GGUF)
# =====================================================================
LLM_MODELS = {
    "qwen-0.5b": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "desc": "مدل فوق‌العاده سبک ۵۰۰ میلیون پارامتری (~۳۹۰ مگابایت) برای دانلود سریع و تست فوری"
    },
    "qwen-1.5b": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "desc": "مدل سبک و بهینه ۱.۵ میلیاردی (~۹۸۰ مگابایت) با کیفیت مناسب روی CPU"
    },
    "qwen-3b": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "desc": "کوئن ۲.۵ سه میلیارد پارامتری (~۱.۹ گیگابایت)، کیفیت عالی برای اجرای یکپارچه"
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
# ۴. انتخاب مدل‌های فعال و شتاب‌دهنده گرافیکی (Active Selection & Hardware)
# =====================================================================
ACTIVE_EMBEDDING = "e5-small"  # مدل امبدینگ فعال: e5-small برای سبکی و سرعت
ACTIVE_LLM = "qwen-1.5b"       # مدل زبانی فعال یکپارچه با llama-cpp (کوئن ۱.۵ میلیارد)

# تنظیمات شتاب‌دهنده کارت گرافیک (GPU Offloading):
# مقدار 1- در llama-cpp به معنی انتقال ۱۰۰٪ لایه‌ها به VRAM کارت گرافیک (مثلاً Nvidia T4 کولب) است.
# در محیط محلی ویندوز بدون کارت گرافیک اختصاصی یا در صورت نبود درایور، روی 0 (فقط CPU) تنظیم می‌شود.
LLM_GPU_LAYERS = int(os.environ.get("LLM_GPU_LAYERS", "-1" if IS_COLAB else "0"))

# دانلود خودکار مدل زبانی در زمان اجرا:
# در کولب (با سرعت اینترنت بالا) فعال است، اما روی سیستم محلی غیرفعال است تا در صورت فیلترینگ یا قطعی نت، برنامه معطل نماند و در حالت Mock بالا بیاید.
AUTO_DOWNLOAD_LLM = os.environ.get("AUTO_DOWNLOAD_LLM", "1" if IS_COLAB else "0") == "1"

# =====================================================================
# ۵. تنظیمات قطعه‌بندی اسناد (Document Chunking Configuration)
# =====================================================================
CHUNK_SIZE = 500       # حداکثر طول هدف هر چانک به کاراکتر
CHUNK_OVERLAP = 100    # میزان همپوشانی چانک‌های مجاور به کاراکتر


def validate_chunk_config(chunk_size: int = None, model_name: str = None) -> bool:
    """
    بررسی اعتبارسنجی اندازه چانک نسبت به حداکثر طول ورودی مدل امبدینگ فعال.
    اگر طول چانک از سقف مجاز مدل بیشتر باشد، خطای صریح توسعه‌دهنده برمی‌گرداند.
    """
    chk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
    chk_overlap = CHUNK_OVERLAP
    mdl_name = model_name if model_name is not None else ACTIVE_EMBEDDING

    if chk_overlap >= chk_size:
        raise ValueError(
            f"[Developer Configuration Error] میزان همپوشانی ({chk_overlap}) "
            f"نمی‌تواند بزرگتر یا مساوی طول چانک ({chk_size}) باشد!"
        )

    if mdl_name not in EMBEDDING_MODELS:
        raise ValueError(f"[Config Error] مدل امبدینگ '{mdl_name}' در تنظیمات EMBEDDING_MODELS یافت نشد.")

    model_info = EMBEDDING_MODELS[mdl_name]
    max_chars = model_info.get("max_chars", 1200)
    max_tokens = model_info.get("max_tokens", 512)

    if chk_size > max_chars:
        raise ValueError(
            f"\n{'='*75}\n"
            f"[خطای پیکربندی توسعه‌دهنده - Developer Configuration Error]\n"
            f"طول چانک انتخابی شما ({chk_size} کاراکتر) از حداکثر ظرفیت ورودی مدل امبدینگ '{mdl_name}' ({max_chars} کاراکتر) بیشتر است!\n"
            f"حداکثر طول مجاز برای مدل '{mdl_name}': {max_chars} کاراکتر (معادل تقریبی {max_tokens} توکن) است.\n"
            f"راهکار: لطفاً CHUNK_SIZE را در فایل config.py به مقداری کمتر یا مساوی {max_chars} کاهش دهید تا متن دچار Truncation نشود.\n"
            f"{'='*75}"
        )
    return True


# اعتبارسنجی خودکار در زمان بارگذاری کانفیگ
validate_chunk_config()

# =====================================================================
# ۶. توابع مدیریت دیتابیس و مدل‌ها (Database & Model Helpers)
# =====================================================================

def save_db_info(model_name: str):
    """
    توضیح:
        ذخیره شناسنامه کامل پایگاه داده برداری داخل فایل db_info.json کنار پایگاه داده.
    ورودی:
        model_name (str): کلید مدل امبدینگ (مثلاً 'e5-small')
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    model_data = EMBEDDING_MODELS.get(model_name, {})
    info = {
        "embedding_model": model_name,
        "dimension": model_data.get("dim", None),
        "distance_metric": "cosine",
        "description": model_data.get("desc", "")
    }
    with open(DB_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

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
    try:
        from vector_db import VectorDB
        vdb = VectorDB()
        deleted_count = vdb.clear_all_collections()
        print(f"[!] کالکشن‌های پایگاه داده برداری ({deleted_count} کالکشن) با موفقیت پاک شدند.")
    except Exception as err:
        if DB_DIR.exists():
            try:
                shutil.rmtree(DB_DIR, ignore_errors=True)
                print("[!] پوشه پایگاه داده برداری (ChromaDB) پاکسازی شد.")
            except Exception as e:
                print(f"[!] خطا در پاکسازی پوشه دیتابیس برداری: {e}")

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

def set_active_llm(new_model_key: str) -> bool:
    """
    توضیح:
        تنظیم مدل زبانی فعال (LLM).
    ورودی:
        new_model_key (str): نام کلید مدل از لیست LLM_MODELS (مثلاً 'qwen-0.5b' یا 'qwen-1.5b')
    """
    global ACTIVE_LLM
    if new_model_key not in LLM_MODELS:
        print(f"[!] مدل زبانی '{new_model_key}' در لیست LLM_MODELS یافت نشد.")
        return False
    ACTIVE_LLM = new_model_key
    print(f"[*] مدل زبانی فعال روی '{new_model_key}' تنظیم شد.")
    return True

def get_active_llm_path() -> Path:
    """مسیر فایل وزن مدل زبانی فعال در سیستم محلی."""
    llm_info = LLM_MODELS.get(ACTIVE_LLM, {})
    filename = llm_info.get("filename", "")
    return LLM_DIR / filename

def get_active_embedding_path() -> Path:
    """مسیر پوشه مدل امبدینگ فعال در سیستم محلی."""
    return EMBEDDING_DIR / ACTIVE_EMBEDDING

def setup_infrastructure():
    """
    توضیح:
        بررسی و آماده‌سازی فایل‌های مدل امبدینگ و مدل زبانی (GGUF محلی) از طریق ModelManager.
    """
    manager = ModelManager(root_dir=MODELS_ROOT)
    return manager.ensure_active_models()

if __name__ == "__main__":
    setup_infrastructure()

