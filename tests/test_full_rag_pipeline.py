"""
tests/test_full_rag_pipeline.py
----------------------------------
آزمون جامع یکپارچگی سراسری هسته سیستم RAG آفلاین (End-to-End Master Pipeline Test).

این فایل تمام ماژول‌های توسعه‌داده‌شده توسط زینب را در یک زنجیره پیوسته تست می‌کند:
  ۱. schemas.py (دریافت چانک‌های رسمی فاطمه و تبدیل به شیء DocumentChunk)
  ۲. config.py (تنظیمات سراسری، شناسنامه دیتابیس و مدیریت مسیرها)
  ۳. model_manager.py (مدیریت پوشه‌ها و وزن‌های مدل‌های محلی)
  ۴. embedder.py (تولید بردارهای معنایی و تزریق پیشوندهای E5)
  ۵. vector_db.py (ذخیره‌سازی و مدیریت کالکشن‌ها در ChromaDB)
  ۶. retriever.py (بازیابی هیبرید معنایی + واژگانی با فیوژن RRF)
  ۷. prompt_builder.py (قالب‌بندی هوشمند اسناد، حافظه تاریخچه و گاردریل ضد توهم)
  ۸. answer_generator.py (استریم کلمات به رابط کاربری، خروج سریع و پیوست استنادها)
  ۹. get_rag_response_stream (ارکستراتور ناهمگام متصل به رابط کاربری مشکات)
"""

import sys
import json
import asyncio
import unittest
from pathlib import Path
from typing import List

# تنظیم کدگذاری خروجی ترمینال روی UTF-8
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
from prompt_builder import PromptBuilder
from answer_generator import AnswerGenerator, get_rag_response_stream


TEST_CHAT_ID = "chat_e2e_master_test"


class TestFullRAGPipeline(unittest.TestCase):
    """
    مجموعه تست‌های یکپارچگی گام‌به‌گام از ورودی چانک‌ها تا تحویل استریم به رابط کاربری.
    """

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 75)
        print("  🚀 شروع آزمون سراسری زنجیره کامل RAG (End-to-End Master Test)")
        print("=" * 75)

        # ۱. آماده‌سازی نمونه چانک‌های استاندارد ورودی (شبیه‌ساز خروجی فاطمه)
        cls.raw_chunks = [
            DocumentChunk(
                chunk_id="doc_hr_c001",
                doc_id="hr_policy.pdf",
                chunk_index=0,
                text="ساعت کاری شرکت از شنبه تا چهارشنبه، از ساعت ۸:۰۰ صبح الی ۱۶:۳۰ بعدازظهر می‌باشد.",
                page_number=2,
                char_start=0,
                char_end=85
            ),
            DocumentChunk(
                chunk_id="doc_hr_c002",
                doc_id="hr_policy.pdf",
                chunk_index=1,
                text="میزان مرخصی استحقاقی سالانه برای کلیه پرسنل تمام‌وقت ۲۶ روز کاری در سال تعیین شده است.",
                page_number=3,
                char_start=86,
                char_end=175
            ),
            DocumentChunk(
                chunk_id="doc_sec_c001",
                doc_id="security_guide.pdf",
                chunk_index=0,
                text="استفاده از حافظه‌های فلش و خروج هرگونه اطلاعات محرمانه از سازمان اکیداً ممنوع بوده و مشمول پیگرد انضباطی است.",
                page_number=1,
                char_start=0,
                char_end=110
            )
        ]

        # ۲. مقداردهی اولیه پایگاه داده برداری و پاکسازی کالکشن تست قبلی
        cls.vdb = VectorDB()
        cls.vdb.delete_chat_collection(TEST_CHAT_ID)

        # ۳. ساخت اشیای هسته سیستم
        cls.embedder = Embedder()
        cls.retriever = Retriever(embedder=cls.embedder, vector_db=cls.vdb)
        cls.prompt_builder = PromptBuilder(max_history_turns=2)
        cls.answer_gen = AnswerGenerator(
            retriever=cls.retriever,
            prompt_builder=cls.prompt_builder,
            mock_mode=True  # استفاده از شبیه‌ساز پایدار برای تست‌های خودکار سریع
        )

    @classmethod
    def tearDownClass(cls):
        # پاکسازی نهایی کالکشن تست از دیسک
        cls.vdb.delete_chat_collection(TEST_CHAT_ID)
        print("\n" + "=" * 75)
        print("  🧹 پاکسازی کالکشن‌های موقت انجام شد و دیتابیس تست بسته شد.")
        print("=" * 75)

    def test_step_1_schemas_and_embedder_flow(self):
        """گام ۱: دریافت چانک‌های فاطمه و تبدیل موفقیت‌آمیز به بردارهای عددی ۴ قسمتی ChromaDB"""
        print("\n[گام ۱] آزمون تبدیل چانک‌های DocumentChunk به داده‌های برداری با Embedder...")

        self.assertEqual(len(self.raw_chunks), 3)

        # تبدیل چانک‌ها به فرمت ۴ قسمتی ChromaDB
        embedded_pack = self.embedder.embed_chunks(self.raw_chunks)

        self.assertIn("ids", embedded_pack)
        self.assertIn("embeddings", embedded_pack)
        self.assertIn("documents", embedded_pack)
        self.assertIn("metadatas", embedded_pack)

        self.assertEqual(len(embedded_pack["ids"]), 3)
        self.assertEqual(len(embedded_pack["embeddings"][0]), self.embedder.dimension)
        print(f"  └─ ۳ چانک با بعد برداری {self.embedder.dimension} با موفقیت تولید شدند.")

        # ذخیره پکیج برداری برای استفاده در گام بعد
        TestFullRAGPipeline.embedded_pack = embedded_pack

    def test_step_2_vectordb_ingestion(self):
        """گام ۲: ایجاد کالکشن چت در ChromaDB و نمایه‌سازی (Indexing) چانک‌ها"""
        print("\n[گام ۲] آزمون ثبت و ایندکس داده‌ها در VectorDB به تفکیک chat_id...")

        col = self.vdb.get_or_create_collection(TEST_CHAT_ID)
        self.assertIsNotNone(col)

        # درج داده‌ها
        inserted_count = self.vdb.add_chunks(TEST_CHAT_ID, self.embedded_pack)
        self.assertEqual(inserted_count, 3)

        # بررسی تعداد چانک‌ها و وضعیت has_chunks
        self.assertEqual(self.vdb.count_chunks(TEST_CHAT_ID), 3)
        self.assertTrue(self.vdb.has_chunks(TEST_CHAT_ID))
        print(f"  └─ ۳ چانک در کالکشن اختصاصی '{TEST_CHAT_ID}' ثبت شد (has_chunks=True).")

    def test_step_3_retriever_hybrid_search(self):
        """گام ۳: جستجوی هیبرید در ریتریور و استخراج مرتبط‌ترین چانک با ذکر شماره صفحه"""
        print("\n[گام ۳] آزمون بازیابی ترکیبی (Dense + BM25 + RRF) برای سوال کارمندی...")

        query = "پرسنل سالانه چقدر حق مرخصی دارند؟"
        result = self.retriever.retrieve(
            query_text=query,
            chat_id=TEST_CHAT_ID,
            top_k=2,
            verbose=False
        )

        self.assertTrue(result["has_relevant_context"])
        self.assertGreater(len(result["top_chunks"]), 0)

        best = result["top_chunks"][0]
        self.assertEqual(best["doc_id"], "hr_policy.pdf")
        self.assertEqual(best["page_number"], 3)
        self.assertIn("۲۶ روز", best["text"])
        print(f"  └─ چانک شماره صفحه ۳ با بالاترین امتیاز بازیابی شد: {best['chunk_id']}")

        # ذخیره نتیجه برای تست پرامپت‌بیلدر
        TestFullRAGPipeline.last_retrieval = result

    def test_step_4_prompt_builder_packaging(self):
        """گام ۴: بسته‌بندی چانک‌ها و سوال در پرامپت‌بیلدر با اعمال قوانین ضد توهم"""
        print("\n[گام ۴] آزمون ساخت پرامپت استاندارد ضد توهم در PromptBuilder...")

        query = "پرسنل سالانه چقدر حق مرخصی دارند؟"
        messages = self.prompt_builder.build_messages(
            query=query,
            retriever_result_or_chunks=self.last_retrieval,
            chat_history=[{"role": "user", "content": "سلام"}, {"role": "assistant", "content": "درود"}]
        )

        # بررسی ترتیب پیام‌ها
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("فقط و فقط", messages[0]["content"])  # قانون طلایی ضد توهم

        last_user_msg = messages[-1]["content"]
        self.assertIn("### زمینه‌ها و اسناد مرجع:", last_user_msg)
        self.assertIn("شماره صفحه: 3", last_user_msg)
        self.assertIn("۲۶ روز کاری", last_user_msg)
        print("  └─ پیام‌ها با ترتیب system -> history -> user با استناد به صفحه ۳ بسته‌بندی شدند.")

    def test_step_5_answer_generator_streaming_and_citations(self):
        """گام ۵: تولید پاسخ استریمی و ارسال مراجع به رابط کاربری با AnswerGenerator"""
        print("\n[گام ۵] آزمون تولید استریم زنده کلمات و استنادها با AnswerGenerator...")

        query = "ساعت کاری شرکت چقدر است؟"
        stream_chunks = list(self.answer_gen.generate_stream(
            query=query,
            chat_id=TEST_CHAT_ID
        ))

        # تفکیک توکن‌ها و مراجع
        tokens = [c["content"] for c in stream_chunks if c.get("type") == "token"]
        sources = [c for c in stream_chunks if c.get("type") == "sources"]

        self.assertGreater(len(tokens), 0)
        self.assertEqual(len(sources), 1)

        citations = sources[0].get("citations", [])
        self.assertGreater(len(citations), 0)

        # بررسی مطابقت شماره صفحه با سند
        top_citation = citations[0]
        self.assertEqual(top_citation["doc_id"], "hr_policy.pdf")
        self.assertEqual(top_citation["page_number"], 2)
        print(f"  └─ استریم موفقیت‌آمیز بود و ارجاع به سند '{top_citation['filename']}' صفحه {top_citation['page_number']} صادر شد.")

    def test_step_6_fast_exit_end_to_end(self):
        """گام ۶: آزمون خروج سریع سراسری در مواجهه با سوال کاملاً پرت و بی‌ربط"""
        print("\n[گام ۶] آزمون گارد خروج سریع در برابر سوال نامربوط («پایتخت کشور ژاپن کجاست؟»)...")

        query = "پایتخت کشور ژاپن کجاست؟"
        stream_chunks = list(self.answer_gen.generate_stream(
            query=query,
            chat_id=TEST_CHAT_ID
        ))

        full_reply = "".join([c["content"] for c in stream_chunks if c.get("type") == "token"]).strip()
        sources = [c for c in stream_chunks if c.get("type") == "sources"][0]

        # بررسی پیام صریح عدم وجود پاسخ
        self.assertIn("پاسخی برای این پرسش در اسناد بارگذاری‌شده شما یافت نشد", full_reply)
        # مراجع باید خالی باشد
        self.assertEqual(sources.get("citations"), [])
        print("  └─ خروج سریع فعال شد و بدون درگیر کردن مدل زبانی، پیام آماده صادر شد.")

    def test_step_7_async_orchestrator_contract(self):
        """گام ۷: آزمون ارکستراتور ناهمگام get_rag_response_stream (مطابق قرارداد فرانت‌اند مشکات)"""
        print("\n[گام ۷] آزمون اجرای ناهمگام get_rag_response_stream متصل به UI...")

        async def run_ui_consumer():
            received_tokens = []
            final_sources = None

            async for chunk in self.answer_gen.get_rag_response_stream(
                query="استفاده از فلش در شرکت مجاز است؟",
                chat_id=TEST_CHAT_ID
            ):
                if chunk["type"] == "token":
                    received_tokens.append(chunk["content"])
                elif chunk["type"] == "sources":
                    final_sources = chunk["citations"]

            return "".join(received_tokens), final_sources

        answer_text, citations = asyncio.run(run_ui_consumer())

        self.assertGreater(len(answer_text), 0)
        self.assertIsNotNone(citations)
        self.assertGreater(len(citations), 0)
        self.assertEqual(citations[0]["doc_id"], "security_guide.pdf")
        self.assertEqual(citations[0]["page_number"], 1)
        print(f"  └─ قرارداد رابط کاربری ۱۰۰٪ پاس شد. پاسخ امنیتی با استناد به صفحه ۱ سند امنیتی استریم گردید.")


def run_full_pipeline_test():
    """اجرای مستقیم فایل تست همراه با گزارش تفصیلی"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFullRAGPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n" + "=" * 75)
        print("  🏆 تبریک زینب عزیز! تمام ۷ مرحله زنجیره کامل RAG با موفقیت ۱۰۰٪ پاس شدند.")
        print("     سیستم از دریافت چانک‌های فاطمه تا تحویل به رابط کاربری مشکات کاملاً هماهنگ است.")
        print("=" * 75 + "\n")
    return result.wasSuccessful()


if __name__ == "__main__":
    run_full_pipeline_test()
