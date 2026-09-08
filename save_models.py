# save_models.py
"""
این اسکریپت رو فقط یه بار (قبل از استفاده آفلاین) اجرا کن.
مدل embedding رو از HuggingFace دانلود کرده و محلی ذخیره می‌کنه.

اجرا:
    python save_models.py
    # یا برای مدل large:
    EMBEDDING_MODEL=large python save_models.py
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ─── import تنظیمات از config ──────────────────────────────────────
from config import EMBEDDING_MODEL_NAME, MODELS_DIR, AVAILABLE_MODELS

def download_and_save(model_name: str, save_path: str) -> None:
    """
    مدل رو از HuggingFace دانلود کرده و در مسیر داده‌شده ذخیره می‌کنه.
    
    Args:
        model_name: نام مدل روی HuggingFace (مثلاً 'intfloat/multilingual-e5-small')
        save_path:  مسیر محلی برای ذخیره‌سازی
    """
    # ساخت پوشه اگه وجود نداره (شامل پوشه‌های میانی هم می‌شه)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    print(f"[↓] دانلود مدل: {model_name}")
    print(f"[→] مسیر ذخیره‌سازی: {save_path}")
    print("    (این عملیات ممکنه چند دقیقه طول بکشه...)\n")

    # دانلود مدل — sentence-transformers همه فایل‌های لازم رو می‌آره
    model = SentenceTransformer(model_name)

    # ذخیره کامل مدل روی دیسک
    model.save(save_path)

    print(f"\n[✓] مدل با موفقیت ذخیره شد: {save_path}")


def main() -> None:
    # تعیین کلید مدل فعال از متغیر محیطی (پیش‌فرض: small)
    model_key = os.environ.get("EMBEDDING_MODEL", "small")
    
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(
            f"مدل '{model_key}' معتبر نیست. "
            f"گزینه‌های موجود: {list(AVAILABLE_MODELS.keys())}"
        )

    model_name = AVAILABLE_MODELS[model_key]["name"]
    
    # هر مدل توی پوشه جداگانه‌ای ذخیره می‌شه
    # مثلاً: models/small/ یا models/large/
    save_path = os.path.join(MODELS_DIR, model_key)

    # اگه قبلاً دانلود شده، دوباره دانلود نمی‌کنه
    if os.path.exists(save_path) and os.listdir(save_path):
        print(f"[✓] مدل '{model_key}' از قبل موجوده: {save_path}")
        print("    برای دانلود مجدد، پوشه رو حذف کن و دوباره اجرا کن.")
        return

    download_and_save(model_name, save_path)


if __name__ == "__main__":
    main()
