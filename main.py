"""
main.py — نقطه ورود اصلی سامانه دستیار هوشمند اسناد PDF (Offline RAG)
---------------------------------------------------------------------
وظایف این ماژول:
  ۱. تنظیم محیط اجرا و پشتیبانی از یونیکد (UTF-8) در ویندوز.
  ۲. آماده‌سازی و اعتبارسنجی اولیه پایگاه‌های داده (SQLite و ChromaDB).
  ۳. اعتبارسنجی هماهنگی طول چانک‌ها با مدل امبدینگ فعال (config.py).
  ۴. راه‌اندازی و بارگذاری اولیه سرویس هماهنگ‌کننده (Orchestrator).
  ۵. اجرای سرور رابط کاربری تعاملی (NiceGUI).
"""

import sys
import os
from pathlib import Path

# ۱. تنظیم خروجی ترمینال روی UTF-8 برای جلوگیری از خطاهای سیستم‌عامل ویندوز
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
import chat_database as db
from orchestrator import get_orchestrator
from ui import start_ui_server


def print_system_banner():
    """چاپ شناسنامه و مشخصات سامانه در کنسول."""
    banner = f"""
{'='*75}
🚀  سامانه دستیار هوشمند اسناد PDF فارسی (RAG کاملاً آفلاین)
{'='*75}
🔹 وضعیت محیط: {'Google Colab' if config.IS_COLAB else 'سیستم محلی (Local)'}
🔹 مدل امبدینگ فعال: {config.ACTIVE_EMBEDDING} (ابعاد: {config.EMBEDDING_MODELS[config.ACTIVE_EMBEDDING]['dim']})
🔹 حداکثر ظرفیت ورودی مدل: {config.EMBEDDING_MODELS[config.ACTIVE_EMBEDDING]['max_tokens']} توکن (~{config.EMBEDDING_MODELS[config.ACTIVE_EMBEDDING]['max_chars']} کاراکتر)
🔹 تنظیمات قطعه‌بندی: طول چانک = {config.CHUNK_SIZE} کاراکتر | همپوشانی = {config.CHUNK_OVERLAP} کاراکتر
🔹 مدل زبانی فعال: {config.ACTIVE_LLM}
🔹 پایگاه داده برداری: {config.DB_DIR}
🔹 پوشه اسناد و چانک‌ها: {config.DATA_DIR}
{'='*75}
🌐 آدرس دسترسی به رابط کاربری: http://localhost:{config.SERVER_PORT}
{'='*75}
"""
    print(banner)


def initialize_system():
    """بررسی پیش‌نیازها و راه‌اندازی زیرساخت‌های برنامه."""
    print("[*] در حال آماده‌سازی و بررسی جداول دیتابیس SQLite...")
    db.initialize_database()

    print("[*] در حال اعتبارسنجی تنظیمات قطعه‌بندی اسناد...")
    config.validate_chunk_config()

    # ساخت پوشه‌های اصلی در صورت عدم وجود
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] در حال آماده‌سازی لایه ارکستراتور و بررسی پایگاه برداری...")
    orch = get_orchestrator()
    print(f"[✓] اتصال به پایگاه برداری ChromaDB برقرار شد.")

    print_system_banner()


def main():
    """نقطه شروع اصلی اجرای پروژه."""
    try:
        initialize_system()
        port = config.SERVER_PORT
        print(f"[*] در حال اجرای وب‌سرور رابط کاربری NiceGUI روی پورت {port}...\n")
        start_ui_server(port=port, reload=False)
    except KeyboardInterrupt:
        print("\n[!] برنامه توسط کاربر متوقف شد.")
    except Exception as e:
        print(f"\n[❌] خطای بحرانی در راه‌اندازی سامانه: {e}")
        raise


if __name__ == "__main__":
    main()
