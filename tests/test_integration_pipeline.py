"""
tests/test_integration_pipeline.py
-------------------------------------------------
تست یکپارچگی (End-to-End Integration Test) برای هسته RAG:
آزمون همزمان و بدون نقض قرارداد بین ماژول‌های:
  - config.py (تنظیمات سراسری، شناسنامه دیتابیس)
  - model_manager.py (مدیریت دیسکی و مسیر مدل‌ها)
  - embedder.py (تولید بردارهای معنایی و پیشوندهای مدل E5)
  - vector_db.py (پایگاه داده برداری ChromaDB)
  - chat_database.py (پایگاه داده متنی SQLite چت‌ها)
  - schemas.py (قرارداد داده‌ای DocumentChunk)
"""

import sys
import json
from pathlib import Path

# تنظیم کدگذاری یونیکد برای ویندوز
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# افزودن ریشه پروژه به مسیر پایتون
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from schemas import DocumentChunk
from embedder import Embedder
from vector_db import VectorDB
import chat_database as chat_db

INTEGRATION_CHAT_ID = "chat_test_integration_e2e"


def test_1_database_passport_sync():
    """
    تست ۱: بررسی شناسنامه پایگاه داده برداری (db_info.json) و انطباق آن با تنظیمات کانفیگ
    """
    print("\n--- [تست ۱] بررسی شناسنامه پایگاه داده برداری (db_info.json) ---")
    assert config.DB_INFO_FILE.exists(), f"فایل شناسنامه در مسیر {config.DB_INFO_FILE} یافت نشد!"

    with open(config.DB_INFO_FILE, "r", encoding="utf-8") as f:
        passport = json.load(f)

    print(f"[✓] شناسنامه لود شد: {passport}")
    assert "embedding_model" in passport, "کلید embedding_model در شناسنامه وجود ندارد!"
    assert passport["embedding_model"] == config.ACTIVE_EMBEDDING, (
        f"مدل داخل شناسنامه ({passport['embedding_model']}) با مدل فعال ({config.ACTIVE_EMBEDDING}) یکسان نیست!"
    )
    assert passport.get("dimension") == config.EMBEDDING_MODELS[config.ACTIVE_EMBEDDING]["dim"]
    assert passport.get("distance_metric") == "cosine"
    print(f"[✓] شناسنامه کاملاً منطبق بر مدل فعال '{config.ACTIVE_EMBEDDING}' است.")


def test_2_embedder_model_manager_integration():
    """
    تست ۲: بارگذاری مدل از طریق Embedder و ModelManager و بررسی ابعاد و صحت بردار
    """
    print("\n--- [تست ۲] آزمون یکپارچگی Embedder و ModelManager ---")
    embedder = Embedder()

    expected_dim = config.EMBEDDING_MODELS[config.ACTIVE_EMBEDDING]["dim"]
    assert embedder.dimension == expected_dim, "ابعاد بردار با کانفیگ همخوانی ندارد!"

    # تست برداری‌سازی پرسش با پیشوند خودکار E5
    query_text = "روش‌های خلاصه‌سازی اسناد متنی فارسی"
    vector = embedder.embed_query(query_text)

    assert isinstance(vector, list), "خروجی بردار پرسش باید لیست پایتونی باشد!"
    assert len(vector) == expected_dim, f"طول بردار باید {expected_dim} باشد اما {len(vector)} است!"

    print(f"[✓] مدل با موفقیت از مسیر لوکال بارگذاری شد.")
    print(f"[✓] پرسش کاربر با موفقیت برداری شد (طول بردار: {len(vector)}).")
    return embedder


def test_3_schemas_to_embedding(embedder: Embedder):
    """
    تست ۳: خواندن چانک‌های واقعی از mock_chunks.json، تبدیل به شیء DocumentChunk و برداری‌سازی
    """
    print("\n--- [تست ۳] آزمون زنجیره چانک‌های schemas -> Embedder ---")
    mock_file = BASE_DIR / "mock_chunks.json"
    assert mock_file.exists(), "فایل mock_chunks.json یافت نشد!"

    with open(mock_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # تبدیل دیکشنری‌ها به اشیاء رسمی DocumentChunk طبق قرارداد
    chunks = [DocumentChunk.from_dict(item) for item in raw_data]
    assert len(chunks) > 0, "لیست چانک‌ها خالی است!"
    print(f"[✓] تعداد {len(chunks)} چانک استاندارد DocumentChunk بارگذاری شد.")

    # تبدیل چانک‌ها به فرمت ۴ قسمتی ChromaDB توسط Embedder
    embedded_data = embedder.embed_chunks(chunks)

    # اعتبارسنجی ساختار ۴ قسمتی
    for key in ["ids", "documents", "embeddings", "metadatas"]:
        assert key in embedded_data, f"کلید الزامی '{key}' در خروجی embed_chunks وجود ندارد!"
        assert len(embedded_data[key]) == len(chunks), f"طول لیست '{key}' با تعداد چانک‌ها برابر نیست!"

    print(f"[✓] بسته ۴ قسمتی داده‌ها با موفقیت توسط امبدر ساخته شد (تعداد: {len(embedded_data['ids'])}).")
    return embedded_data


def test_4_vector_db_population(embedded_data: dict):
    """
    تست ۴: ایجاد کالکشن خالی و سپس ذخیره‌سازی داده‌های واقعی در VectorDB
    """
    print("\n--- [تست ۴] آزمون ذخیره‌سازی و چرخه کالکشن در VectorDB ---")
    vdb = VectorDB()

    # ۱. اطمینان از پاک بودن کالکشن قبل از تست
    vdb.delete_chat_collection(INTEGRATION_CHAT_ID)

    # ۲. ایجاد کالکشن خالی و بررسی وضعیت (سناریوی تصمیم تیم)
    col = vdb.get_or_create_collection(INTEGRATION_CHAT_ID)
    assert vdb.count_chunks(INTEGRATION_CHAT_ID) == 0
    assert vdb.has_chunks(INTEGRATION_CHAT_ID) is False
    print(f"[✓] کالکشن خالی '{col.name}' با موفقیت ساخته شد و has_chunks=False است.")

    # ۳. درج چانک‌های واقعی برداری‌شده
    inserted = vdb.add_chunks(INTEGRATION_CHAT_ID, embedded_data)
    assert inserted == len(embedded_data["ids"])
    assert vdb.count_chunks(INTEGRATION_CHAT_ID) == len(embedded_data["ids"])
    assert vdb.has_chunks(INTEGRATION_CHAT_ID) is True
    print(f"[✓] تعداد {inserted} چانک واقعی در پایگاه داده برداری ذخیره شدند (has_chunks=True).")

    return vdb


def test_5_end_to_end_semantic_search(embedder: Embedder, vdb: VectorDB):
    """
    تست ۵: تست طلایی RAG - پرسش معنایی و بازیابی مرتبط‌ترین چانک + تست فیلتر سند
    """
    print("\n--- [تست ۵] آزمون طلایی: جستجوی شباهت معنایی واقعی (Semantic Retrieval) ---")

    # سوال در رابطه با خلاصه‌سازی است که در سند doc_0002 وجود دارد:
    query = "رویکردهای خلاصه‌سازی اسناد چیست؟"
    query_vector = embedder.embed_query(query)

    # جستجو در دیتابیس برداری
    results = vdb.query_raw(INTEGRATION_CHAT_ID, query_vector, top_k=2)

    top_id = results["ids"][0][0]
    top_doc = results["documents"][0][0]
    top_distance = results["distances"][0][0]
    top_metadata = results["metadatas"][0][0]

    print(f"[✓] سوال کاربر: «{query}»")
    print(f"[✓] رتبه ۱ بازیابی‌شده: شناسه '{top_id}' از سند '{top_metadata['doc_id']}'")
    print(f"[✓] فاصله کسینوسی: {top_distance:.4f} (هر چه به ۰ نزدیک‌تر، شباهت بیشتر)")
    print(f"[✓] بخشی از متن بازیابی‌شده: {top_doc[:65]}...")

    # اعتبارسنجی معنایی: نتیجه اول باید حتماً مربوط به خلاصه‌سازی (doc_0002) باشد
    assert "خلاصه‌سازی" in top_doc or top_metadata["doc_id"] == "doc_0002", (
        "جستجوی معنایی چانک مرتبط با خلاصه‌سازی را در رتبه ۱ نیاورده است!"
    )

    # تست فیلتر سند: مجبور کردن سیستم به جستجو فقط در doc_0001
    filtered_results = vdb.query_raw(INTEGRATION_CHAT_ID, query_vector, top_k=2, filter_doc_id="doc_0001")
    for meta in filtered_results["metadatas"][0]:
        assert meta["doc_id"] == "doc_0001", "فیلتر doc_id کار نکرده و چانکی از سند دیگر بازگشته است!"

    print(f"[✓] فیلتر سند (doc_id='doc_0001') با موفقیت مانع نفوذ سایر اسناد شد.")


def test_6_chat_database_and_cleanup_sync(vdb: VectorDB):
    """
    تست ۶: یکپارچگی پایگاه داده چت‌ها (SQLite) با پایگاه داده برداری (ChromaDB) و حذف‌ها
    """
    print("\n--- [تست ۶] آزمون هماهنگی VectorDB و chat_database ---")

    # ثبت یک چت آزمایشی در SQLite
    chat_db.initialize_database()
    sample_chat_id = chat_db.create_chat(title="چت آزمایشی یکپارچگی")
    assert sample_chat_id is not None
    print(f"[✓] رکورد چت شماره {sample_chat_id} در SQLite ساخته شد.")

    # افزودن پیام و فایل آزمایشی
    chat_db.add_message(sample_chat_id, role="user", content="سلام این یک تست است")
    chat_db.add_file(sample_chat_id, "contract.pdf", "/fake/path/contract.pdf", status="ready")

    # تست حذف فایل doc_0002 از کالکشن برداری
    initial_chunks = vdb.count_chunks(INTEGRATION_CHAT_ID)
    vdb.delete_file_chunks(INTEGRATION_CHAT_ID, doc_id="doc_0002")
    after_del_chunks = vdb.count_chunks(INTEGRATION_CHAT_ID)
    assert after_del_chunks < initial_chunks, "چانک‌های doc_0002 حذف نشدند!"
    print(f"[✓] چانک‌های doc_0002 از دیتابیس برداری حذف شدند ({initial_chunks} -> {after_del_chunks}).")

    # تست پاکسازی کامل کالکشن چت تست
    vdb.delete_chat_collection(INTEGRATION_CHAT_ID)
    assert vdb.count_chunks(INTEGRATION_CHAT_ID) == 0
    print(f"[✓] کالکشن تست {INTEGRATION_CHAT_ID} به طور کامل از دیسک پاک شد.")

    # تست تابع پاکسازی رکوردهای چت
    chat_db.clear_chat_records()
    assert len(chat_db.get_chats()) == 0, "رکوردهای چت پاکسازی نشدند!"
    print(f"[✓] متد clear_chat_records با موفقیت رکوردهای پایگاه داده چت را پاک کرد.")


def run_pipeline_test():
    """
    اجرای کل زنجیره تست یکپارچگی هسته RAG
    """
    print("=" * 70)
    print("   🚀 آغاز آزمون یکپارچگی کامل هسته سیستم RAG (End-to-End Pipeline)")
    print("=" * 70)

    test_1_database_passport_sync()
    embedder = test_2_embedder_model_manager_integration()
    embedded_data = test_3_schemas_to_embedding(embedder)
    vdb = test_4_vector_db_population(embedded_data)
    test_5_end_to_end_semantic_search(embedder, vdb)
    test_6_chat_database_and_cleanup_sync(vdb)

    print("\n" + "=" * 70)
    print("  🏆 فوق‌العاده است زینب عزیز! تمام ماژول‌های امبدر، پایگاه داده برداری،")
    print("     شناسنامه، مدل منیجر و چت دیتابیس در هماهنگی ۱۰۰٪ کامل کار می‌کنند.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_pipeline_test()
