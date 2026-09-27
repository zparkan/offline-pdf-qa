"""
tests/test_integration_pipeline.py
-------------------------------------------------
تست یکپارچگی سراسری (End-to-End Core RAG Pipeline Test):
آزمون جریان کامل داده‌ها بین تمامی ماژول‌های هسته سیستم:
  ۱. config.py (تنظیمات سراسری، شناسنامه دیتابیس)
  ۲. model_manager.py (مدیریت مسیرها و بارگذاری مدل محلی)
  ۳. embedder.py (تولید بردارهای معنایی و پیشوندهای E5)
  ۴. schemas.py (قرارداد رسمی DocumentChunk)
  ۵. vector_db.py (پایگاه داده برداری ChromaDB)
  ۶. retriever.py (موتور بازیابی ترکیبی Dense + BM25 + RRF + اعتبارسنجی ارتباط)
  ۷. chat_database.py (پایگاه داده SQLite تاریخچه چت‌ها)
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
from retriever import Retriever
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

    # ۲. ایجاد کالکشن خالی و بررسی وضعیت
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


def test_5_full_retriever_pipeline(embedder: Embedder, vdb: VectorDB):
    """
    تست ۵: تست طلایی RAG یکپارچه با ماژول بازیاب (Retriever)
    شامل بازیابی ترکیبی (Dense + BM25 + RRF)، فیلتر اسناد و تشخیص سوال نامربوط
    """
    print("\n--- [تست ۵] آزمون طلایی: بازیابی ترکیبی کامل با ماژول Retriever ---")
    retriever = Retriever(embedder=embedder, vector_db=vdb)

    # ۵-۱. تست بازیابی معنایی و واژگانی سوال مرتبط با خلاصه‌سازی
    query = "رویکردهای متفاوت خلاصه‌سازی اسناد بلند چیست؟"
    res1 = retriever.retrieve(query, chat_id=INTEGRATION_CHAT_ID, top_k=3, verbose=True)

    top_chunks = res1["top_chunks"]
    assert len(top_chunks) > 0, "هیچ چانکی بازیابی نشد!"
    assert res1["has_relevant_context"] is True, "سیستم به اشتباه سوال مرتبط را نامرتبط تشخیص داد!"

    best_chunk = top_chunks[0]
    print(f"[✓] بهترین نتیجه: شناسه '{best_chunk['chunk_id']}' از سند '{best_chunk['doc_id']}' (صفحه {best_chunk['page_number']})")
    print(f"[✓] امتیاز RRF: {best_chunk['score']} | شباهت معنایی: {best_chunk['similarity']}")
    assert best_chunk["doc_id"] == "doc_0002", "انتظار می‌رفت سند خلاصه‌سازی doc_0002 در رتبه ۱ باشد!"

    # ۵-۲. تست فیلتر اسناد: اجبار سیستم به جستجو فقط در سند doc_0001
    print("\n[تست ۵-۲] فیلتر اسناد: جستجو فقط در doc_0001...")
    res_filtered = retriever.retrieve(
        query_text=query,
        chat_id=INTEGRATION_CHAT_ID,
        top_k=2,
        filter_doc_ids="doc_0001",
        verbose=False
    )
    for c in res_filtered["top_chunks"]:
        assert c["doc_id"] == "doc_0001", f"خطا: فیلتر سند نقض شد و سند {c['doc_id']} برگردانده شد!"
    print(f"[✓] فیلتر اسناد با موفقیت اعمال شد و تمام چانک‌ها متعلق به doc_0001 بودند.")

    # ۵-۳. تست تشخیص سوال کاملاً نامربوط (عدم وجود پاسخ در منابع)
    print("\n[تست ۵-۳] اعتبارسنجی سوال نامربوط: «طرز تهیه کیک شکلاتی خانگی در فر چیست؟»...")
    res_irrelevant = retriever.retrieve(
        query_text="طرز تهیه کیک شکلاتی خانگی در فر چیست؟",
        chat_id=INTEGRATION_CHAT_ID,
        top_k=2,
        min_similarity=0.80,
        verbose=False
    )
    assert res_irrelevant["has_relevant_context"] is False, "خطا: سوال نامربوط نباید دارای زمینه مرتبط باشد!"
    print(f"[✓] سوال نامربوط با موفقیت شناسایی شد (has_relevant_context: False | بیشترین شباهت: {res_irrelevant['max_similarity']}).")


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
    test_5_full_retriever_pipeline(embedder, vdb)
    test_6_chat_database_and_cleanup_sync(vdb)

    print("\n" + "=" * 70)
    print("  🏆 تبریک! تمام ماژول‌های هسته سیستم (Config، ModelManager، Embedder،")
    print("     VectorDB، Retriever و ChatDatabase) در هماهنگی ۱۰۰٪ کار می‌کنند.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_pipeline_test()
