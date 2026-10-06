"""
orchestrator.py — ماژول رهبر و هماهنگ‌کننده مرکزی (Orchestrator Service)
---------------------------------------------------------------------
مسئولیت:
  ۱. پیوند کامل بین ماژول‌های پیش‌پردازش (فاطمه)، هسته RAG (زینب) و رابط کاربری/دیتابیس (مشکات).
  ۲. مدیریت چرخه حیات اسناد (Document Ingestion Pipeline):
     استخراج متن ➔ پالایش و نرمال‌سازی ➔ قطعه‌بندی هوشمند ➔ امبدینگ ➔ ثبت در ChromaDB.
  ۳. ذخیره گام‌به‌گام فایل‌های میانی در پوشه اختصاصی گفتگو (data/{chat_id}/).
  ۴. مدیریت چرخه پرسش و پاسخ (QA Stream Pipeline):
     نرمال‌سازی سوال ➔ دریافت تاریخچه چت ➔ بازیابی ترکیبی ➔ استریم توکن‌ها و استخراج مراجع.
  ۵. هماهنگی رویدادهای حذف سند و حذف گفتگو میان دیسک، SQLite و ChromaDB.
"""

import sys
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, AsyncGenerator

# تنظیم خروجی ترمینال ویندوز روی UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
import chat_database as db
from schemas import DocumentChunk, ExtractedDocument, PreprocessedDocument
from pdf_extractor import extract_pdf, save_extracted_document
from text_preprocessor import clean_persian_text, preprocess_document, save_preprocessed_document
from chunk_manager import chunk_document, save_chunks_to_json

from embedder import Embedder
from vector_db import VectorDB
from retriever import Retriever
from prompt_builder import PromptBuilder
from answer_generator import AnswerGenerator


class Orchestrator:
    """
    کلاس مدیریت و رهبری کل پایپ‌لاین سیستم پرسش و پاسخ اسناد.
    """

    def __init__(
        self,
        vector_db: Optional[VectorDB] = None,
        embedder: Optional[Embedder] = None,
        retriever: Optional[Retriever] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        answer_generator: Optional[AnswerGenerator] = None,
    ):
        """
        مقداردهی اولیه و تزریق وابستگی‌های سراسری.
        """
        self.vector_db = vector_db or VectorDB()
        self.embedder = embedder or Embedder()
        self.retriever = retriever or Retriever(embedder=self.embedder, vector_db=self.vector_db)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.answer_generator = answer_generator or AnswerGenerator(
            retriever=self.retriever,
            prompt_builder=self.prompt_builder
        )

    def process_document(
        self,
        file_path: Union[str, Path],
        chat_id: Union[int, str],
        filename: str,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """
        اجرای کامل خط لوله پردازش سند از PDF خام تا ذخیره در پایگاه برداری.
        تمامی خروجی‌های میانی مستقیماً در پوشه data/{chat_id}/ ذخیره می‌شوند.

        ورودی:
          - file_path: مسیر فایل اصلی PDF در دیسک
          - chat_id: شناسه گفتگوی فعال
          - filename: نام فایل سند
          - on_progress: تابع کال‌بک جهت دریافت پیام و درصد پیشرفت

        خروجی:
          - bool: موفقیت یا عدم موفقیت
        """
        pdf_file = Path(file_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"فایل در مسیر مشخص‌شده یافت نشد: {pdf_file}")

        chat_dir = db.get_chat_data_dir(chat_id)
        stem_name = Path(filename).stem

        # ۱. مرحله استخراج متن صفحات (F-02 فاطمه)
        extracted_doc = extract_pdf(pdf_file, source_type="real", on_progress=on_progress)
        extracted_path = chat_dir / f"{stem_name}_extracted.json"
        save_extracted_document(extracted_doc, extracted_path)

        # ۲. مرحله پالایش و نرمال‌سازی متن (F-03 فاطمه)
        clean_doc = preprocess_document(extracted_doc, on_progress=on_progress)
        clean_path = chat_dir / f"{stem_name}_clean.json"
        save_preprocessed_document(clean_doc, clean_path)

        # ۳. مرحله قطعه‌بندی هوشمند متن (F-04)
        chunks = chunk_document(clean_doc, on_progress=on_progress)
        chunks_path = chat_dir / f"{stem_name}_chunks.json"
        save_chunks_to_json(chunks, chunks_path)

        if not chunks:
            if on_progress:
                on_progress("هشدار: متنی در سند برای قطعه‌بندی یافت نشد.", 1.00)
            return True

        # ۴. مرحله تولید بردارها (زینب)
        if on_progress:
            on_progress(f"تولید بردارها با مدل هوش مصنوعی ({len(chunks)} چانک)...", 0.80)
        
        chunk_data = self.embedder.embed_chunks(chunks)

        # ۵. مرحله نمایه‌سازی و ذخیره در ChromaDB (زینب)
        if on_progress:
            on_progress("نمایه‌سازی در پایگاه داده برداری...", 0.95)

        self.vector_db.add_chunks(chat_id=chat_id, chunk_data=chunk_data)

        # ۶. اعلام اتمام موفقیت‌آمیز
        if on_progress:
            on_progress("تکمیل پردازش اسناد", 1.00)

        return True

    async def ask_question_stream(
        self,
        query: str,
        chat_id: Union[int, str],
        filter_doc_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        خط لوله پردازش سوال کاربر و تحویل زنده جریان پاسخ (Streaming).

        مراحل:
          ۱. پاکسازی و نرمال‌سازی سوال کاربر.
          ۲. خواندن تاریخچه چت از SQLite مشکات.
          ۳. بازیابی هیبرید + پرامپت استنادی + تولید استریم با مدل زبانی.
        """
        # ۱. نرمال‌سازی متن سوال با ماژول فاطمه
        normalized_query = clean_persian_text(query).strip()
        if not normalized_query:
            normalized_query = query.strip()

        # ۲. واکشی تاریخچه پیام‌های چت
        raw_messages = db.get_messages(chat_id)
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in raw_messages
            if m.get("content")
        ]

        # ۳. دریافت استریم از پاسخ‌ساز RAG
        async for packet in self.answer_generator.get_rag_response_stream(
            query=normalized_query,
            chat_id=chat_id,
            filter_doc_ids=filter_doc_ids,
            chat_history=chat_history
        ):
            yield packet

    def delete_document(self, chat_id: Union[int, str], filename: str, file_id: int) -> bool:
        """
        حذف کامل یک سند از تمام بخش‌های سیستم:
        ۱. حذف از دیسک (PDF اصلی و فایل‌های JSON میانی)
        ۲. حذف بردارها از ChromaDB
        ۳. حذف رکورد از دیتابیس SQLite
        """
        chat_dir = db.get_chat_data_dir(chat_id)
        stem_name = Path(filename).stem

        # پاکسازی فایل‌های دیسک در پوشه چت
        patterns = [filename, f"{stem_name}_extracted.json", f"{stem_name}_clean.json", f"{stem_name}_chunks.json"]
        for pat in patterns:
            f_p = chat_dir / pat
            if f_p.exists():
                try:
                    f_p.unlink()
                except OSError:
                    pass

        # پاکسازی چانک‌های برداری از ChromaDB
        # با توجه به اینکه doc_id بر مبنای هش یا نام فایل است، بر اساس نام سند حذف می‌شود
        self.vector_db.delete_file_chunks(chat_id=chat_id, doc_id=filename)

        # پاکسازی از جدول فایل‌های SQLite
        db.delete_file(file_id)
        return True

    def delete_chat(self, chat_id: Union[int, str]) -> bool:
        """
        حذف کامل یک گفتگو از تمام بخش‌های سیستم:
        ۱. پاکسازی پوشه data/{chat_id}/ از دیسک
        ۲. حذف کالکشن چت از ChromaDB
        ۳. حذف رکورد گفتگو و تمام پیام‌ها از دیتابیس SQLite
        """
        chat_dir = db.get_chat_data_dir(chat_id)
        if chat_dir.exists():
            try:
                shutil.rmtree(chat_dir, ignore_errors=True)
            except OSError:
                pass

        # حذف کالکشن اختصاصی در کرومادی‌بی
        self.vector_db.delete_chat_collection(chat_id)

        # حذف از پایگاه داده چت‌ها
        db.delete_chat(chat_id)
        return True


# نمونه اشتراکی سراسری (Singleton)
_orchestrator_instance: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """دریافت نمونه اشتراکی ارکستراتور."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()
    return _orchestrator_instance
