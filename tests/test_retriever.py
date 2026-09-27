"""
tests/test_retriever.py
----------------------------------
مجموعه آزمون‌های واحد و یکپارچه‌سازی ماژول بازیاب ترکیبی (Retriever).

سناریوهای مورد آزمون:
  ۱. جستجوی دقیق واژگانی (کد پیگیری یا شماره ماده - برتری BM25).
  ۲. جستجوی مفهومی و معنایی (مترادف‌ها و عدم تطابق کلمه دقیق - برتری Dense).
  ۳. فیلتر اسناد (اطمینان از واکشی چانک‌ها فقط از سندهای مجاز).
  ۴. مدیریت حالات مرزی (چت خالی یا عدم وجود چانک).
"""

import sys
from pathlib import Path

# افزودن ریشه پروژه به مسیر پایتون برای دسترسی آسان به ماژول‌ها
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from schemas import DocumentChunk
from embedder import Embedder
from vector_db import VectorDB
from retriever import Retriever


def setup_test_data(chat_id: str, embedder: Embedder, vdb: VectorDB):
    """
    ایجاد چند چانک آزمایشی با ویژگی‌های واژگانی و مفهومی متمایز در کالکشن تست.
    """
    vdb.delete_chat_collection(chat_id)

    chunks = [
        DocumentChunk(
            chunk_id="doc1_c001",
            doc_id="contract_101",
            chunk_index=0,
            text="طرفین قرارداد موظفند در تاریخ یکم مهر ماه نسبت به تسویه حساب مالی به مبلغ ۵۰ میلیون تومان اقدام نمایند.",
            page_number=1,
            char_start=0,
            char_end=105
        ),
        DocumentChunk(
            chunk_id="doc1_c002",
            doc_id="contract_101",
            chunk_index=1,
            text="در صورت بروز شرایط اضطراری یا فورس‌ماژور، تعهدات پیمانکار به مدت سی روز معلق خواهد شد و مسئولیتی نخواهد داشت.",
            page_number=2,
            char_start=106,
            char_end=220
        ),
        DocumentChunk(
            chunk_id="doc2_c001",
            doc_id="rules_202",
            chunk_index=0,
            text="کد پیگیری پرونده اداری شماره ۹۸۲۳۴-ب می‌باشد و بایستی در تمامی نامه‌ها قید گردد.",
            page_number=4,
            char_start=0,
            char_end=85
        ),
        DocumentChunk(
            chunk_id="doc2_c002",
            doc_id="rules_202",
            chunk_index=1,
            text="حق فسخ قرارداد تنها در صورت نقض صریح بندهای ایمنی کار توسط کارفرما اعمال می‌گردد.",
            page_number=5,
            char_start=86,
            char_end=175
        )
    ]

    embedded = embedder.embed_chunks(chunks)
    vdb.add_chunks(chat_id, embedded)
    return chunks


def run_all_tests():
    print("=" * 65)
    print("🚀 آغاز اجرای آزمون‌های خودکار ماژول Retriever")
    print("=" * 65)

    test_chat = "test_retriever_suite_chat"
    embedder = Embedder()
    vdb = VectorDB()

    try:
        setup_test_data(test_chat, embedder, vdb)
        retriever = Retriever(embedder=embedder, vector_db=vdb)

        # -------------------------------------------------------------
        # سناریوی ۱: جستجوی کد خاص (تست قدرت BM25 + ترکیب RRF)
        # -------------------------------------------------------------
        print("\n[تست ۱] جستجوی کد پیگیری خاص «۹۸۲۳۴-ب»...")
        res1 = retriever.retrieve("کد پیگیری ۹۸۲۳۴-ب چیست؟", chat_id=test_chat, top_k=2, verbose=False)
        top_chunks_1 = res1["top_chunks"]

        assert len(top_chunks_1) > 0, "خطا: هیچ چانکی یافت نشد!"
        assert top_chunks_1[0]["chunk_id"] == "doc2_c001", f"خطا: انتظار doc2_c001 بود اما {top_chunks_1[0]['chunk_id']} آمد!"
        assert "۹۸۲۳۴-ب" in top_chunks_1[0]["text"], "خطا: کد پیگیری در متن چانک برتر نبود!"
        print(f"  └─ نتیجه: چانک {top_chunks_1[0]['chunk_id']} با موفقیت در رتبه اول قرار گرفت (امتیاز RRF: {top_chunks_1[0]['score']}). [تایید شد ✓]")

        # -------------------------------------------------------------
        # سناریوی ۲: جستجوی معنایی و مفهومی (تست قدرت Dense + RRF)
        # -------------------------------------------------------------
        print("\n[تست ۲] جستجوی مفهومی «حوادث غیرمترقبه» (مترادف با فورس‌ماژور)...")
        res2 = retriever.retrieve("اگر حوادث غیرمترقبه رخ دهد وضعیت تعهدات چیست؟", chat_id=test_chat, top_k=2, verbose=False)
        top_chunks_2 = res2["top_chunks"]

        assert len(top_chunks_2) > 0, "خطا: هیچ چانکی یافت نشد!"
        assert top_chunks_2[0]["chunk_id"] == "doc1_c002", f"خطا: انتظار doc1_c002 بود اما {top_chunks_2[0]['chunk_id']} آمد!"
        print(f"  └─ نتیجه: چانک مفهوم فورس‌ماژور ({top_chunks_2[0]['chunk_id']}) در رتبه اول بازیابی شد. [تایید شد ✓]")

        # -------------------------------------------------------------
        # سناریوی ۳: فیلتر کردن سند خاص
        # -------------------------------------------------------------
        print("\n[تست ۳] فیلتر اسناد: جستجو فقط در سند contract_101...")
        res3 = retriever.retrieve(
            query_text="شرایط لغو یا فسخ قرارداد چیست؟",
            chat_id=test_chat,
            top_k=2,
            filter_doc_ids="contract_101",
            verbose=False
        )
        top_chunks_3 = res3["top_chunks"]

        assert len(top_chunks_3) > 0, "خطا: چانکی برگردانده نشد!"
        for ch in top_chunks_3:
            assert ch["doc_id"] == "contract_101", f"خطا: فیلتر نقض شد! سند نامعتبر {ch['doc_id']} برگردانده شد."
        print(f"  └─ نتیجه: تمام {len(top_chunks_3)} چانک فقط از سند contract_101 انتخاب شدند. [تایید شد ✓]")

        # -------------------------------------------------------------
        # سناریوی ۴: حالت مرزی چت خالی
        # -------------------------------------------------------------
        print("\n[تست ۴] حالت مرزی: جستجو در چت خالی که فایلی ندارد...")
        res4 = retriever.retrieve("سوال تستی؟", chat_id="empty_nonexistent_chat", top_k=3, verbose=False)
        assert res4["top_chunks"] == [], "خطا: برای چت خالی باید لیست خالی برگردد!"
        assert res4["stats"]["total_chunks_available"] == 0, "خطا: تعداد چانک‌ها باید صفر باشد!"
        assert res4["has_relevant_context"] is False, "خطا: برای چت خالی باید has_relevant_context برابر False باشد!"
        print("  └─ نتیجه: پاسخ خالی ایمن بدون هیچ‌گونه پرتاب خطا دریافت شد. [تایید شد ✓]")

        # -------------------------------------------------------------
        # سناریوی ۵: سوال کاملاً نامربوط به اسناد (تشخیص عدم وجود پاسخ در اسناد)
        # -------------------------------------------------------------
        print("\n[تست ۵] سوال نامربوط: «طرز تهیه کیک شکلاتی خانگی در فر چیست؟»...")
        res5 = retriever.retrieve(
            query_text="طرز تهیه کیک شکلاتی خانگی در فر چیست؟",
            chat_id=test_chat,
            top_k=2,
            min_similarity=0.80,
            verbose=False
        )
        assert res5["has_relevant_context"] is False, "خطا: سیستم سوال نامربوط را مرتبط تشخیص داد!"
        print(f"  └─ نتیجه: سیستم با موفقیت تشخیص داد که پاسخ در اسناد نیست (has_relevant_context: False | بیشترین شباهت: {res5['max_similarity']}). [تایید شد ✓]")

        print("\n" + "=" * 65)
        print("🎉 تمام ۵ آزمون با موفقیت ۱۰۰٪ پاس شدند!")
        print("=" * 65)

    finally:
        # پاکسازی کالکشن تست
        vdb.delete_chat_collection(test_chat)


if __name__ == "__main__":
    run_all_tests()
