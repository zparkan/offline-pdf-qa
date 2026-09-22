"""
generate_synthetic_data.py
----------------------------------
تولید داده‌ی مصنوعی (Mock Data) مطابق قرارداد داده‌ای پروژه (data_contract.md).

هدف: زینب (embedding/FAISS) و مشکات (LLM/prompt) بتوانند بدون نیاز به
      استخراج واقعی PDF (که هنوز کامل نشده) روی داده‌ای با ساختار دقیقاً
      یکسان با خروجی نهایی، توسعه و تست بدهند.

خروجی سه فایل JSON در پوشه‌ی sample_data/:
  - documents.json  -> سطح ۱: خروجی استخراج متن (شبیه‌سازی F-02)
  - chunks.json      -> سطح ۲: خروجی چانک‌بندی (تحویل اصلی به زینب)
  - qa_pairs.json     -> سطح ۳: دیتاست ارزیابی (پیش‌نویس F-05/F-06)
"""

import json
import os
import random

from schemas import DocumentChunk, QAPair

random.seed(42)  # برای تکرارپذیری نتایج بین اعضای تیم

OUTPUT_DIR = "sample_data"

# ----------------------------------------------------------------------
# ۱. چند متن نمونه‌ی فارسی (جایگزین متن واقعی PDF تا زمان تکمیل F-02)
#    این‌ها را می‌توان بعداً با محتوای واقعی مقالات/گزارش‌های جمع‌آوری‌شده
#    در مرحله‌ی دوم پروژه (بخش ۷ پروپوزال) جایگزین کرد.
# ----------------------------------------------------------------------
SAMPLE_DOCS = {
    "doc_0001": {
        "filename": "gozaresh_fanni_namoone.pdf",
        "pages": {
            1: "این گزارش به بررسی روش‌های پردازش زبان طبیعی فارسی می‌پردازد. "
               "هدف اصلی، ارائه‌ی یک چارچوب عملی برای استخراج اطلاعات از اسناد اداری است. "
               "نویسنده در ابتدا پیشینه‌ی موضوع را مرور می‌کند.",
            2: "در بخش دوم، نویسنده استدلال می‌کند که مدل‌های سبک‌وزن برای اجرای آفلاین "
               "مناسب‌ترند، زیرا محدودیت سخت‌افزاری کاربران نهایی را رعایت می‌کنند. "
               "این استدلال بر پایه‌ی مقایسه‌ی مصرف حافظه در سه مدل مختلف بنا شده است.",
            3: "نتیجه‌گیری گزارش نشان می‌دهد که ترکیب چانک‌بندی همپوشان با مدل‌های "
               "embedding چندزبانه، دقت بازیابی را نسبت به روش پایه به میزان محسوسی افزایش می‌دهد.",
        },
    },
    "doc_0002": {
        "filename": "maghale_elmi_namoone.pdf",
        "pages": {
            1: "این مقاله دو رویکرد متفاوت برای خلاصه‌سازی اسناد بلند را مقایسه می‌کند: "
               "روش استخراجی و روش تولیدی.",
            2: "نتایج آزمایش‌ها نشان می‌دهد روش تولیدی در حفظ پیوستگی معنایی برتری دارد، "
               "اما روش استخراجی از نظر سرعت اجرا مناسب‌تر سامانه‌های آفلاین است.",
        },
    },
}

QUESTION_TYPES = ["مستقیم", "تفسیری", "مقایسه‌ای"]


def build_documents():
    """سطح ۱: ساخت خروجی استخراج متن (شبیه‌ساز F-02)."""
    documents = []
    for doc_id, info in SAMPLE_DOCS.items():
        pages = [
            {"page_number": p, "raw_text": text}
            for p, text in sorted(info["pages"].items())
        ]
        documents.append(
            {
                "doc_id": doc_id,
                "filename": info["filename"],
                "source_type": "synthetic",
                "pages": pages,
            }
        )
    return documents


def simple_chunk_text(text, chunk_size=80, overlap=20):
    """
    نسخه‌ی ساده‌شده‌ی چانک‌بندی همپوشان (کاراکتر-محور) — صرفاً برای تولید
    داده‌ی مصنوعی. نسخه‌ی نهایی F-04 از روش Recursive Character/Sentence-based
    استفاده خواهد کرد و دقیق‌تر است.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append((start, end, text[start:end]))
        if end == len(text):
            break
        start = end - overlap  # همپوشانی
    return chunks


def build_chunks(documents):
    """
    سطح ۲: ساخت خروجی چانک‌بندی (تحویل اصلی به زینب).

    خروجی یک لیست تخت (flat list) از چانک‌هاست، نه گروه‌بندی‌شده بر اساس سند —
    چون ایندکس FAISS یکپارچه است و پیمایش مستقیم روی یک جریان تخت را ساده‌تر می‌کند.
    هر چانک فیلد doc_id خودش را دارد تا اطلاعات تعلق به سند از دست نرود.

    نکته‌ی مهم: این تابع عمداً فیلدهای "vector" و "score" را تولید نمی‌کند.
    - vector: باید توسط ماژول embedding (زینب) از روی متن این چانک‌ها ساخته شود.
    - score: مقداری وابسته به سوال کاربر است و فقط در زمان اجرای واقعی retrieval
      معنا دارد؛ نمی‌تواند ویژگی ثابت یک چانک باشد.
    """
    all_chunks = []
    for doc in documents:
        doc_id = doc["doc_id"]
        idx = 0
        for page in doc["pages"]:
            for char_start, char_end, chunk_text in simple_chunk_text(page["raw_text"]):
                chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_c{idx:03d}",
                    doc_id=doc_id,
                    chunk_index=idx,
                    text=chunk_text.strip(),
                    page_number=page["page_number"],
                    char_start=char_start,
                    char_end=char_end,
                    token_count=len(chunk_text.split()),
                )
                all_chunks.append(chunk)
                idx += 1
    return all_chunks


def build_qa_pairs(flat_chunks, doc_ids):
    """
    پیش‌نویس ساختاری برای دیتاست ارزیابی (نمونه‌ی اولیه‌ی F-06).
    توجه: این خروجی صرفاً برای تست ساختار JSON است — سوال‌ها و پاسخ‌های
    placeholder هستند و دیتاست واقعی F-06 باید با ۵۰ جفت سوال-جواب دستی
    و مستند از اسناد واقعی ساخته شود، نه با این تابع.
    """
    qa_pairs = []
    for doc_id in doc_ids:
        chunk_ids = [c.chunk_id for c in flat_chunks if c.doc_id == doc_id]
        for i in range(3):  # سه سوال نمونه به ازای هر سند، صرفا برای تست ساختار
            qtype = QUESTION_TYPES[i % len(QUESTION_TYPES)]
            support = random.sample(chunk_ids, k=min(2, len(chunk_ids)))
            qa_pairs.append(
                QAPair(
                    qa_id=f"{doc_id}_qa{i+1:03d}",
                    doc_id=doc_id,
                    question=f"[نمونه‌ی جای‌گذاری‌شده] سوال {qtype} شماره {i+1} درباره‌ی {doc_id}",
                    question_type=qtype,
                    answer_reference="[نمونه‌ی جای‌گذاری‌شده] پاسخ مرجع باید در F-06 توسط انسان نوشته شود.",
                    supporting_chunk_ids=support,
                )
            )
    return qa_pairs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    documents = build_documents()
    flat_chunks = build_chunks(documents)
    doc_ids = [d["doc_id"] for d in documents]
    qa_pairs = build_qa_pairs(flat_chunks, doc_ids)

    with open(os.path.join(OUTPUT_DIR, "documents.json"), "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    # تحویلی اصلی F-05: mock_chunks.json — بر اساس کلاس مشترک DocumentChunk
    with open(os.path.join(OUTPUT_DIR, "mock_chunks.json"), "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in flat_chunks], f, ensure_ascii=False, indent=2)

    # پیش‌نویس ساختاری برای F-06 (نه تحویلی نهایی)
    with open(os.path.join(OUTPUT_DIR, "qa_pairs_draft.json"), "w", encoding="utf-8") as f:
        json.dump([qa.to_dict() for qa in qa_pairs], f, ensure_ascii=False, indent=2)

    print(f"ساخته شد: {len(documents)} سند، {len(flat_chunks)} چانک، {len(qa_pairs)} جفت سوال-جواب (پیش‌نویس).")
    print(f"فایل‌ها در پوشه‌ی '{OUTPUT_DIR}/' ذخیره شدند. تحویلی اصلی F-05: mock_chunks.json")


if __name__ == "__main__":
    main()
