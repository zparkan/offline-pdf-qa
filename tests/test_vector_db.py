"""
tests/test_vector_db.py
-------------------------------------------------
تست‌های اعتبارسنجی ماژول پایگاه داده برداری (VectorDB)
شامل:
  ۱. تست راه‌اندازی کلاینت و پوشه پایدار ChromaDB
  ۲. تست ساخت کالکشن خالی برای چت جدید و انطباق متریک کسینوسی (Cosine)
  ۳. تست ذخیره داده‌های ۴ قسمتی و رفتار امن upsert
  ۴. تست پرس‌وجوی برداری خام (query_raw) و فرمت خروجی
  ۵. تست فیلتر کردن چانک‌ها بر اساس آیدی سند (filter_doc_id)
  ۶. تست حذف چانک‌های یک فایل خاص بدون آسیب به سایر فایل‌های چت
  ۷. تست حذف کامل کالکشن چت و رفتار ضد خطا (Fail-Safe)
"""

import sys
from pathlib import Path

# تنظیم خروجی یونیکد برای ویندوز
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# افزودن ریشه پروژه به sys.path جهت ایمپورت مستقیم ماژول‌ها
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from vector_db import VectorDB

# آیدی آزمایشی مجزا برای ایزوله ماندن محیط تست
TEST_CHAT_ID = "test_unit_99"


def test_1_client_initialization():
    """
    تست ۱: بررسی راه‌اندازی صحیح کلاینت ChromaDB و وجود پوشه ذخیره‌سازی
    """
    print("\n--- [تست ۱] بررسی راه‌اندازی کلاینت پایگاه داده برداری ---")
    vdb = VectorDB()
    assert vdb.persist_dir.exists(), "پوشه دیتابیس برداری ایجاد نشده است!"
    assert vdb.client is not None, "کلاینت ChromaDB راه‌اندازی نشده است!"
    print(f"[✓] کلاینت با موفقیت ایجاد شد.")
    print(f"[✓] مسیر پایگاه داده برداری: {vdb.persist_dir}")


def test_2_empty_collection_and_cosine_metric():
    """
    تست ۲: بررسی ساخت کالکشن خالی برای چت بدون فایل (سناریوی تصمیم تیم)
    و بررسی تنظیم صحیح متریک فاصله روی Cosine
    """
    print("\n--- [تست ۲] بررسی کالکشن خالی و متریک کسینوسی ---")
    vdb = VectorDB()

    # حذف احتمالی کالکشن تست از اجراهای قبلی
    vdb.delete_chat_collection(TEST_CHAT_ID)

    # ایجاد کالکشن خالی برای چت جدید
    col = vdb.get_or_create_collection(TEST_CHAT_ID)

    assert vdb.count_chunks(TEST_CHAT_ID) == 0, "تعداد چانک‌های کالکشن تازه ساخته‌شده باید صفر باشد!"
    assert vdb.has_chunks(TEST_CHAT_ID) is False, "متد has_chunks برای کالکشن خالی باید False برگرداند!"

    # بررسی متریک فاصله (باید حتماً cosine باشد)
    space_metric = col.metadata.get("hnsw:space")
    assert space_metric == "cosine", f"متریک فاصله باید cosine باشد اما {space_metric} است!"

    print(f"[✓] کالکشن خالی '{col.name}' با موفقیت ساخته شد.")
    print(f"[✓] تعداد چانک‌ها: {vdb.count_chunks(TEST_CHAT_ID)} | has_chunks: {vdb.has_chunks(TEST_CHAT_ID)}")
    print(f"[✓] متریک فضای جستجو: {space_metric}")


def test_3_add_chunks_and_upsert():
    """
    تست ۳: ذخیره چانک‌های ۴ قسمتی (شبیه‌ساز خروجی embedder) و بررسی رفتار upsert
    """
    print("\n--- [تست ۳] ذخیره داده‌های ۴ قسمتی و بررسی upsert ---")
    vdb = VectorDB()

    mock_chunks = {
        "ids": ["doc1_c001", "doc1_c002", "doc2_c001"],
        "documents": [
            "ماده ۱: شرایط عمومی قرارداد استخدام تمام‌وقت",
            "ماده ۲: تعهدات کارفرما و پرداخت حق بیمه تامین اجتماعی",
            "سرفصل ۳: شرایط کاری پروژه‌ای و ساعات دورکاری"
        ],
        "embeddings": [
            [0.1] * 384,
            [0.2] * 384,
            [0.85] * 384
        ],
        "metadatas": [
            {"doc_id": "doc1", "page_number": 1, "filename": "employment.pdf"},
            {"doc_id": "doc1", "page_number": 2, "filename": "employment.pdf"},
            {"doc_id": "doc2", "page_number": 1, "filename": "remote_policy.pdf"}
        ]
    }

    # ذخیره در کالکشن
    saved_count = vdb.add_chunks(TEST_CHAT_ID, mock_chunks)
    assert saved_count == 3, f"باید ۳ چانک ذخیره می‌شد اما {saved_count} چانک ذخیره شد!"
    assert vdb.count_chunks(TEST_CHAT_ID) == 3, "شمارنده چانک‌ها باید عدد ۳ را نشان دهد!"
    assert vdb.has_chunks(TEST_CHAT_ID) is True, "اکنون has_chunks باید True باشد!"

    # تست upsert: ارسال مجدد همان داده‌ها نباید خطا دهد و نباید تعداد را دو برابر کند
    resaved_count = vdb.add_chunks(TEST_CHAT_ID, mock_chunks)
    assert resaved_count == 3
    assert vdb.count_chunks(TEST_CHAT_ID) == 3, "رفتار upsert نقض شده و رکوردهای تکراری اضافه شده است!"

    print(f"[✓] ۳ چانک ۴ قسمتی با موفقیت ذخیره شدند.")
    print(f"[✓] رفتار upsert بررسی شد (بدون خطای شناسه تکراری و حفظ تعداد ۳ چانک).")


def test_4_query_raw():
    """
    تست ۴: پرس‌وجوی برداری خام و بررسی ساختار خروجی برای ماژول Retriever
    """
    print("\n--- [تست ۴] بررسی پرس‌وجوی برداری خام (query_raw) ---")
    vdb = VectorDB()

    # بردار پرسش شبیه به چانک‌های doc1
    query_vector = [0.12] * 384

    results = vdb.query_raw(TEST_CHAT_ID, query_vector, top_k=2)

    assert "ids" in results and "documents" in results and "distances" in results
    returned_ids = results["ids"][0]
    returned_docs = results["documents"][0]
    returned_distances = results["distances"][0]

    assert len(returned_ids) == 2, f"انتظار دریافت ۲ نتیجه برتر داشتیم اما {len(returned_ids)} مورد دریافت شد."
    assert "doc1_c001" in returned_ids, "چانک با بیشترین شباهت (doc1_c001) در نتایج پیدا نشد!"

    print(f"[✓] تعداد نتایج بازگشتی: {len(returned_ids)}")
    print(f"[✓] شناسه‌های برتر: {returned_ids}")
    print(f"[✓] نزدیک‌ترین فاصله کسینوسی محاسبه‌شده: {returned_distances[0]:.4f}")


def test_5_query_with_doc_filter():
    """
    تست ۵: بررسی فیلتر کردن هوشمند بر اساس فایل خاص (ویژگی اختیاری قرارداد)
    """
    print("\n--- [تست ۵] بررسی فیلتر کردن جستجو بر اساس سند (doc_id) ---")
    vdb = VectorDB()

    query_vector = [0.12] * 384  # برداری که در حالت عادی doc1 را برمی‌گرداند

    # ولی ما سیستم را مجبور می‌کنیم فقط در doc2 جستجو کند
    filtered_results = vdb.query_raw(TEST_CHAT_ID, query_vector, top_k=2, filter_doc_id="doc2")
    filtered_ids = filtered_results["ids"][0]

    assert len(filtered_ids) == 1, "تنها باید ۱ چانک متعلق به doc2 برگردد!"
    assert filtered_ids[0] == "doc2_c001", "نتیجه فیلترشده منطبق با doc2 نیست!"

    print(f"[✓] جستجو با فیلتر 'doc2' با موفقیت فقط چانک‌های همان سند را برگرداند: {filtered_ids}")


def test_6_delete_file_chunks():
    """
    تست ۶: حذف چانک‌های متعلق به یک فایل و عدم تغییر بقیه فایل‌های چت
    """
    print("\n--- [تست ۶] بررسی حذف چانک‌های یک فایل خاص ---")
    vdb = VectorDB()

    # وضعیت قبل از حذف: ۳ چانک (۲ تا برای doc1 و ۱ برای doc2)
    assert vdb.count_chunks(TEST_CHAT_ID) == 3

    # حذف تمام چانک‌های doc1
    success = vdb.delete_file_chunks(TEST_CHAT_ID, doc_id="doc1")
    assert success is True, "عملیات حذف فایل با شکست مواجه شد!"

    # وضعیت بعد از حذف: باید دقیقاً ۱ چانک (مربوط به doc2) باقی بماند
    remaining_count = vdb.count_chunks(TEST_CHAT_ID)
    assert remaining_count == 1, f"باید ۱ چانک باقی می‌ماند اما {remaining_count} چانک ماند!"

    # اطمینان از اینکه چانک باقی‌مانده واقعاً doc2 است
    remaining = vdb.query_raw(TEST_CHAT_ID, [0.0] * 384, top_k=5)
    assert remaining["ids"][0] == ["doc2_c001"]

    print(f"[✓] چانک‌های فایل doc1 با موفقیت حذف شدند.")
    print(f"[✓] تعداد چانک‌های باقی‌مانده در گفتگو: {remaining_count} (فقط چانک doc2)")


def test_7_delete_chat_collection():
    """
    تست ۷: حذف کامل کالکشن چت و تست رفتار ضد خطا (Fail-Safe)
    """
    print("\n--- [تست ۷] بررسی حذف کامل کالکشن چت و رفتار امن ---")
    vdb = VectorDB()

    # حذف کالکشن چت
    deleted = vdb.delete_chat_collection(TEST_CHAT_ID)
    assert deleted is True, "حذف کالکشن باید موفقیت‌آمیز باشد."
    assert vdb.count_chunks(TEST_CHAT_ID) == 0, "پس از حذف، تعداد چانک‌ها باید صفر باشد."
    assert vdb.has_chunks(TEST_CHAT_ID) is False

    # تست حذف مجدد یک چت ناموجود (نباید هیچ خطایی رخ دهد)
    safe_delete = vdb.delete_chat_collection("chat_non_existent_12345")
    assert safe_delete is True, "حذف کالکشن ناموجود نباید باعث بروز خطا شود."

    print(f"[✓] کالکشن چت با موفقیت حذف شد.")
    print(f"[✓] رفتار ضد خطا در حذف چت ناموجود تأیید شد.")


def run_all_tests():
    """
    اجرای ترتیبی تمام تست‌های ماژول پایگاه داده برداری
    """
    print("=" * 65)
    print("      شروع آزمون‌های خودکار ماژول پایگاه داده برداری (VectorDB)")
    print("=" * 65)

    test_1_client_initialization()
    test_2_empty_collection_and_cosine_metric()
    test_3_add_chunks_and_upsert()
    test_4_query_raw()
    test_5_query_with_doc_filter()
    test_6_delete_file_chunks()
    test_7_delete_chat_collection()

    print("\n" + "=" * 65)
    print("  🎉 تبریک زینب عزیز! تمام ۷ آزمون پایگاه داده برداری پاس شدند.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_all_tests()
