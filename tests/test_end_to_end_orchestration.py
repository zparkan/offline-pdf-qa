"""
test_end_to_end_orchestration.py
--------------------------------
تست جامع سراسری برای اعتبارسنجی کارکرد ارکستراتور، جریان داده و اتصال ماژول‌ها.
"""

import sys
import asyncio
from pathlib import Path

# UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# افزودن مسیر ریشه پروژه به sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import chat_database as db
from schemas import ExtractedDocument, ExtractedPage
from pdf_extractor import save_extracted_document
from orchestrator import get_orchestrator


async def run_test():
    print("[1] مقداردهی اولیه سیستم و ارکستراتور...")
    db.initialize_database()
    orch = get_orchestrator()

    test_chat_id = "test_chat_999"
    chat_dir = db.get_chat_data_dir(test_chat_id)
    print(f"    پوشه تست چت: {chat_dir}")

    # ایجاد یک سند تستی در پوشه چت
    test_pdf_name = "sample_test.pdf"
    fake_pdf_path = chat_dir / test_pdf_name
    fake_pdf_path.write_text("dummy pdf binary content for test", encoding="utf-8")

    # ساخت فایل اکسترکت شده برای شبیه‌سازی دقیق F-02
    extracted_doc = ExtractedDocument(
        doc_id="doc_sample_test",
        filename=test_pdf_name,
        source_type="synthetic",
        pages=[
            ExtractedPage(
                page_number=1,
                raw_text="هوش مصنوعی شاخه‌ای از علوم کامپیوتر است که به ساخت سیستم‌های هوشمند می‌پردازد.\nاین سیستم‌ها می‌توانند یادگیری ماشینی و پردازش زبان طبیعی را انجام دهند."
            ),
            ExtractedPage(
                page_number=2,
                raw_text="سیستم‌های RAG اسناد متنی را استخراج کرده و به قطعات کوچک تبدیل می‌کنند.\nسپس مدل‌های زبانی بر اساس این اسناد به سوالات پاسخ می‌دهند."
            )
        ]
    )
    extracted_json_path = chat_dir / f"{fake_pdf_path.stem}_extracted.json"
    save_extracted_document(extracted_doc, extracted_json_path)
    print(f"    [✓] فایل استخراج شده ذخیره شد: {extracted_json_path.name}")

    # تست پاکسازی و پیش‌پردازش (F-03)
    from text_preprocessor import preprocess_document, save_preprocessed_document
    clean_doc = preprocess_document(extracted_doc)
    clean_json_path = chat_dir / f"{fake_pdf_path.stem}_clean.json"
    save_preprocessed_document(clean_doc, clean_json_path)
    print(f"    [✓] فایل پالایش شده ذخیره شد: {clean_json_path.name}")

    # تست چانک‌بندی (F-04)
    from chunk_manager import chunk_document, save_chunks_to_json
    chunks = chunk_document(clean_doc)
    chunks_json_path = chat_dir / f"{fake_pdf_path.stem}_chunks.json"
    save_chunks_to_json(chunks, chunks_json_path)
    print(f"    [✓] فایل چانک‌ها ذخیره شد: {chunks_json_path.name} (تعداد: {len(chunks)})")

    # تست ذخیره در ChromaDB
    chunk_data = orch.embedder.embed_chunks(chunks)
    orch.vector_db.add_chunks(test_chat_id, chunk_data)
    chunk_count = orch.vector_db.count_chunks(test_chat_id)
    print(f"    [✓] چانک‌ها با موفقیت در ChromaDB ذخیره شدند (موجودی چت: {chunk_count})")
    assert chunk_count == len(chunks), "عدم تطابق تعداد چانک‌های ذخیره‌شده!"

    # تست استریم پاسخ سوال
    print("\n[2] تست استریم پرسش و پاسخ...")
    test_query = "سیستم‌های RAG چگونه کار می‌کنند؟"
    full_answer = ""
    citations = []

    async for packet in orch.ask_question_stream(test_query, test_chat_id):
        if packet.get("type") == "token":
            full_answer += packet.get("content", "")
        elif packet.get("type") == "sources":
            citations = packet.get("citations", [])

    print(f"    پاسخ دریافتی: {full_answer[:80]}...")
    print(f"    تعداد استنادها: {len(citations)}")
    assert len(full_answer) > 0, "پاسخ مدل نباید خالی باشد!"

    # تست پاکسازی و حذف چت
    print("\n[3] تست حذف کامل چت و پاکسازی...")
    orch.delete_chat(test_chat_id)
    assert not chat_dir.exists(), "پوشه چت باید از دیسک پاک شده باشد!"
    assert orch.vector_db.count_chunks(test_chat_id) == 0, "کالکشن چت باید از ChromaDB پاک شده باشد!"
    print("    [✓] چت با موفقیت از دیسک و ChromaDB پاکسازی شد.")

    print("\n" + "="*60)
    print("🎉 تمامی تست‌های یکپارچه‌سازی با موفقیت ۱۰۰٪ پاس شدند!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_test())
