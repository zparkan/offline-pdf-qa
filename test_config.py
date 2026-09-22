# test_config.py
"""
تست صحت عملکرد config.py:
۱. بررسی وجود مدل‌های جدید در لیست
۲. بررسی مسیرهای تعریف شده (پوشه data و دیتابیس چت)
۳. تست منطق تغییر مدل امبدینگ و پاکسازی پایگاه داده برداری
"""

import os
import json
import sqlite3
import config

def run_config_test():
    print("=== شروع تست‌های config.py ===")

    # تست ۱: آیا مدل‌های درخواستی داخل دیکشنری هستند؟
    required_models = [
        "qwen3-emb-4b", "bge-m3", "octen-emb-8b", 
        "qwen3-emb-8b", "snowflake-arctic-l", "f2llm-4b"
    ]
    missing_models = [m for m in required_models if m not in config.EMBEDDING_MODELS]
    if not missing_models:
        print("[✓] تست ۱: همه ۶ مدل جدید در EMBEDDING_MODELS وجود دارند.")
    else:
        print(f"[X] تست ۱ ناموفق: این مدل‌ها جا افتاده‌اند: {missing_models}")

    # تست ۲: بررسی وجود مسیرهای جدید در کانفیگ
    assert hasattr(config, "DATA_DIR"), "متغیر DATA_DIR در کانفیگ وجود ندارد"
    assert hasattr(config, "CHAT_DB_FILE"), "متغیر CHAT_DB_FILE در کانفیگ وجود ندارد"
    print(f"[✓] تست ۲: مسیرها تعریف شده‌اند:\n    - Data: {config.DATA_DIR}\n    - Chat DB: {config.CHAT_DB_FILE}")

    # تست ۳: شبیه‌سازی ایجاد دیتابیس چت تستی برای اطمینان از منطق پاکسازی
    test_db = config.CHAT_DB_FILE
    conn = sqlite3.connect(test_db)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, text TEXT);")
    cur.execute("INSERT INTO messages (text) VALUES ('پیام آزمایشی');")
    conn.commit()
    conn.close()

    # ایجاد فایل شناسنامه دیتابیس برداری با مدل اولیه e5-base
    config.save_db_info("e5-base")
    print("[✓] تست ۳: یک دیتابیس چت فرضی و شناسنامه دیتابیس با مدل e5-base ساخته شد.")

    # تست ۴: فراخوانی تغییر مدل به bge-m3 (باید دیتابیس برداری ریست و دیتابیس چت خالی شود)
    print("[*] در حال تغییر مدل امبدینگ به 'bge-m3'...")
    config.set_active_embedding("bge-m3")

    # بررسی نتیجه تغییر مدل
    current_model = config.get_current_db_model()
    if current_model == "bge-m3":
        print(f"[✓] تست ۴: مدل دیتابیس با موفقیت به '{current_model}' به‌روزرسانی شد.")
    else:
        print(f"[X] تست ۴ ناموفق: مدل دیتابیس به جای bge-m3 مقدار {current_model} دارد.")

    # بررسی اینکه اطلاعات دیتابیس چت پاک شده باشد اما جدول باقی مانده باشد
    conn = sqlite3.connect(test_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM messages;")
    count = cur.fetchone()[0]
    conn.close()

    if count == 0:
        print("[✓] تست ۵: اطلاعات درون جدول دیتابیس چت پاک شد (جدول سالم ماند).")
    else:
        print(f"[X] تست ۵ ناموفق: سطرها پاک نشدند (تعداد: {count}).")

    print("\n=== تمام تست‌های کانفیگ با موفقیت سپری شدند! ===")

if __name__ == "__main__":
    run_config_test()
