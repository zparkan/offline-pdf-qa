"""
tests/test_embedder.py
-------------------------------------------------
تست‌های اعتبارسنجی ماژول امبدر (Embedder)
شامل:
  ۱. تست بارگذاری مدل و تطابق ابعاد بردار با config.py
  ۲. تست پیشوندهای اختصاصی مدل‌های e5 (passage: / query:)
  ۳. تست تبدیل چانک‌های DocumentChunk به فرمت استاندارد ChromaDB
  ۴. تست برداری کردن سوال کاربر (embed_query)
  ۵. تست هوشمندی و شباهت معنایی بردارها (Semantic Similarity Check)
"""

import sys
from pathlib import Path
import numpy as np

# افزودن ریشه پروژه به sys.path جهت ایمپورت مستقیم ماژول‌ها
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from schemas import DocumentChunk
from embedder import Embedder


def test_1_initialization_and_dimension():
    """
    تست ۱: بررسی لود شدن مدل امبدینگ پیش‌فرض و انطباق ابعاد بردار با config.py
    """
    print("\n--- [تست ۱] بررسی لود شدن مدل و انطباق ابعاد بردار ---")
    embedder = Embedder()
    expected_dim = config.EMBEDDING_MODELS[config.ACTIVE_EMBEDDING]["dim"]

    assert embedder.dimension == expected_dim, (
        f"ابعاد ثبت شده در شیء ({embedder.dimension}) با کانفیگ ({expected_dim}) یکسان نیست!"
    )
    print(f"[✓] مدل فعال: {embedder.model_key}")
    print(f"[✓] ابعاد مورد انتظار: {expected_dim} | ابعاد شیء: {embedder.dimension}")


def test_2_e5_formatting():
    """
    تست ۲: بررسی اضافه شدن خودکار پیشوندهای passage: و query: برای مدل‌های خانواده e5
    """
    print("\n--- [تست ۲] بررسی پیشوندهای مدل E5 ---")
    embedder = Embedder()

    sample_text = "شرکت متعهد به پرداخت مبالغ قرارداد است."
    passage_formatted = embedder._format_text_for_embedding(sample_text, is_query=False)
    query_formatted = embedder._format_text_for_embedding(sample_text, is_query=True)

    if embedder.is_e5_family:
        assert passage_formatted.startswith("passage: "), "پیشوند passage: اضافه نشده است!"
        assert query_formatted.startswith("query: "), "پیشوند query: اضافه نشده است!"
        print(f"[✓] فرمت متن چانک: '{passage_formatted[:25]}...'")
        print(f"[✓] فرمت متن پرسش: '{query_formatted[:25]}...'")
    else:
        assert passage_formatted == sample_text, "برای مدل‌های غیر e5 نباید پیشوند اضافه شود."
        print(f"[✓] مدل غیر e5 است و پیشوندی اضافه نشد.")


def test_3_embed_chunks_output_format():
    """
    تست ۳: تست تابع embed_chunks با چانک‌های شبیه‌سازی‌شده فاطمه بر اساس قرارداد داده‌ای
    """
    print("\n--- [تست ۳] بررسی خروجی embed_chunks و سازگاری با ChromaDB ---")
    embedder = Embedder()

    # ساخت ۲ چانک نمونه مطابق کلاس رسمی DocumentChunk
    sample_chunks = [
        DocumentChunk(
            chunk_id="doc1_c001",
            doc_id="doc1",
            chunk_index=0,
            text="مبلغ کل این قرارداد ده میلیون تومان است که طی دو قسط پرداخت می‌شود.",
            page_number=1,
            char_start=0,
            char_end=68
        ),
        DocumentChunk(
            chunk_id="doc1_c002",
            doc_id="doc1",
            chunk_index=1,
            text="طرف دوم متعهد به انجام خدمات فنی در موعد مقرر یک‌ماهه می‌باشد.",
            page_number=2,
            char_start=69,
            char_end=130
        )
    ]

    result = embedder.embed_chunks(sample_chunks)

    # ۱. بررسی وجود هر ۴ کلید مورد نیاز ChromaDB
    required_keys = {"ids", "documents", "embeddings", "metadatas"}
    assert required_keys.issubset(result.keys()), f"کلیدهای خروجی ناقص است: {result.keys()}"

    # ۲. بررسی تعداد رکوردها
    assert len(result["ids"]) == 2
    assert len(result["documents"]) == 2
    assert len(result["embeddings"]) == 2
    assert len(result["metadatas"]) == 2

    # ۳. بررسی ابعاد بردار تولید شده برای هر چانک
    dim = len(result["embeddings"][0])
    assert dim == embedder.dimension, f"بعد بردار تولید شده ({dim}) با بعد مدل ({embedder.dimension}) همخوانی ندارد!"

    # ۴. بررسی نگهداری درست متادیتا
    assert result["metadatas"][0]["page_number"] == 1
    assert result["metadatas"][0]["doc_id"] == "doc1"
    assert result["ids"][0] == "doc1_c001"

    print(f"[✓] هر ۴ کلید استاندارد ChromaDB تولید شدند.")
    print(f"[✓] تعداد چانک‌ها: {len(result['ids'])} عدد | بعد بردارها: {dim}")
    print(f"[✓] متادیتای شماره صفحه و نام سند به‌درستی نگهداری شدند.")


def test_4_embed_query():
    """
    تست ۴: تست تابع embed_query برای پرسش کاربر
    """
    print("\n--- [تست ۴] بررسی تابع embed_query برای سوال کاربر ---")
    embedder = Embedder()

    query = "هزینه و مبلغ قرارداد چقدر است؟"
    vector = embedder.embed_query(query)

    assert isinstance(vector, list), "خروجی embed_query باید یک لیست پایتونی از floatها باشد."
    assert len(vector) == embedder.dimension, f"بعد بردار سوال ({len(vector)}) با مدل همخوانی ندارد!"
    assert all(isinstance(val, float) for val in vector[:5]), "المان‌های بردار باید float باشند."

    print(f"[✓] سوال کاربر با موفقیت برداری شد.")
    print(f"[✓] طول بردار سوال: {len(vector)} | نمونه مقادیر اول: {[round(x, 4) for x in vector[:3]]}")


def test_5_semantic_similarity_sanity_check():
    """
    تست ۵: تست شباهت معنایی (Semantic Sanity Check)
    سوال مالی باید به چانک مالی شباهت بالاتری نسبت به چانک قوانین مرخصی داشته باشد.
    """
    print("\n--- [تست ۵] آزمون هوشمندی و درک شباهت معنایی ---")
    embedder = Embedder()

    chunk_financial = DocumentChunk(
        chunk_id="fin_01",
        doc_id="doc_fin",
        chunk_index=0,
        text="کلیه هزینه‌ها، دستمزد پرسنل و مالیات پروژه به عهده پیمانکار خواهد بود.",
        page_number=3,
        char_start=0,
        char_end=68
    )

    chunk_leave = DocumentChunk(
        chunk_id="hr_01",
        doc_id="doc_hr",
        chunk_index=0,
        text="مرخصی استحقاقی کارمندان دو روز و نیم در هر ماه کاری محاسبه می‌گردد.",
        page_number=5,
        char_start=0,
        char_end=67
    )

    # برداری کردن هر دو چانک
    chunks_out = embedder.embed_chunks([chunk_financial, chunk_leave])
    v_fin = np.array(chunks_out["embeddings"][0])
    v_hr = np.array(chunks_out["embeddings"][1])

    # برداری کردن سوال با موضوع کاملاً مالی
    query = "میزان دستمزد و هزینه‌های مربوط به مالیات چگونه است؟"
    v_query = np.array(embedder.embed_query(query))

    # محاسبه شباهت کسینوسی (چون بردارها نرمال شده‌اند، ضرب داخلی معادل کسینوس است)
    sim_fin = float(np.dot(v_query, v_fin))
    sim_hr = float(np.dot(v_query, v_hr))

    print(f"  - شباهت سوال مالی با چانک «مالی و دستمزد»: {sim_fin:.4f}")
    print(f"  - شباهت سوال مالی با چانک «مرخصی کارمندان»: {sim_hr:.4f}")

    assert sim_fin > sim_hr, (
        f"خطای منطق معنایی! شباهت با چانک غیرمرتبط ({sim_hr}) بیشتر از چانک مرتبط ({sim_fin}) است!"
    )
    print(f"[✓] مدل با موفقیت تشخیص داد که سوال مالی به چانک مالی بسیار نزدیک‌تر است (اختلاف: {sim_fin - sim_hr:.4f}).")


if __name__ == "__main__":
    print("==================================================")
    print("🚀 شروع آزمون‌های خودکار اعتبارسنجی ماژول Embedder")
    print("==================================================")

    test_1_initialization_and_dimension()
    test_2_e5_formatting()
    test_3_embed_chunks_output_format()
    test_4_embed_query()
    test_5_semantic_similarity_sanity_check()

    print("\n==================================================")
    print("🎉 تمام ۵ آزمون با موفقیت ۱۰۰٪ پاس شدند! ماژول امبدر کاملاً سالم و آماده اتصال است.")
    print("==================================================")
