"""
tests/test_answer_generator.py
----------------------------------
مجموعه آزمون‌های واحد و یکپارچه برای ماژول answer_generator.py.

موارد تحت آزمون:
  ۱. مقداردهی اولیه و سوئیچ امن به حالت شبیه‌ساز (Mock Mode).
  ۲. عملکرد گارد خروج سریع (Fast Exit) در سوالات نامربوط (عدم ارسال به LLM و ارسال پیام پیش‌فرض).
  ۳. استریم زنده توکن‌ها و انطباق با قرارداد StreamChunk فرانت‌اند.
  ۴. استخراج صحیح استنادها و مراجع (Citations) در انتهای پاسخ.
  ۵. خروجی یکپارچه تابع generate_response.
  ۶. اجرای ناهمگام تابع get_rag_response_stream (مطابق قرارداد با مشکات).
"""

import sys
import asyncio
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# افزودن ریشه پروژه به مسیر پایتون
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from answer_generator import AnswerGenerator, get_rag_response_stream
from prompt_builder import PromptBuilder, FALLBACK_NO_CONTEXT_MESSAGE


class TestAnswerGenerator(unittest.TestCase):
    def setUp(self):
        # ساخت ماک برای ریتریور جهت ایزوله‌سازی آزمون‌ها
        self.mock_retriever = MagicMock()
        self.prompt_builder = PromptBuilder()

        # چانک‌های نمونه مرتبط
        self.sample_retrieval_success = {
            "query_text": "سقف مرخصی سالانه چند روز است؟",
            "top_chunks": [
                {
                    "chunk_id": "hr_c001",
                    "text": "بر اساس ماده ۳ اساسنامه، سقف مرخصی سالانه ۲۶ روز کاری است.",
                    "doc_id": "hr_policy.pdf",
                    "page_number": 4,
                    "similarity": 0.92,
                    "score": 0.033
                }
            ],
            "has_relevant_context": True,
            "max_similarity": 0.92,
            "elapsed_ms": 5.0
        }

        # نمونه بازیابی نامرتبط
        self.sample_retrieval_irrelevant = {
            "query_text": "پایتخت کشور کانادا چیست؟",
            "top_chunks": [],
            "has_relevant_context": False,
            "max_similarity": 0.1,
            "elapsed_ms": 3.0
        }

    def test_initialization_and_mock_fallback(self):
        """آزمون: مقداردهی اولیه و فعال‌سازی امن mock_mode بدون کرش کردن سیستم"""
        generator = AnswerGenerator(
            retriever=self.mock_retriever,
            prompt_builder=self.prompt_builder,
            mock_mode=True
        )
        self.assertTrue(generator.mock_mode)
        self.assertIsNotNone(generator.retriever)
        self.assertIsNotNone(generator.prompt_builder)

    def test_fast_exit_on_irrelevant_query(self):
        """آزمون: گارد خروج سریع در سوالات نامربوط (تولید فوری پیام عدم وجود پاسخ و بدون مراجع)"""
        self.mock_retriever.retrieve.return_value = self.sample_retrieval_irrelevant

        generator = AnswerGenerator(
            retriever=self.mock_retriever,
            prompt_builder=self.prompt_builder,
            mock_mode=True
        )

        stream = list(generator.generate_stream(
            query="پایتخت کشور کانادا چیست؟",
            chat_id="test_chat_1"
        ))

        # توکن‌ها باید حاوی پیام پیش‌فرض عدم وجود پاسخ باشند
        tokens = [item["content"] for item in stream if item.get("type") == "token"]
        combined_text = "".join(tokens).strip()

        self.assertIn("پاسخی برای این پرسش در اسناد بارگذاری‌شده شما یافت نشد", combined_text)

        # آخرین آیتم استریم باید مراجع خالی باشد
        last_item = stream[-1]
        self.assertEqual(last_item.get("type"), "sources")
        self.assertEqual(last_item.get("citations"), [])

    def test_generate_stream_tokens_and_citations(self):
        """آزمون: استریم زنده کلمات و ارسال استنادهای دقیق در انتهای استریم"""
        self.mock_retriever.retrieve.return_value = self.sample_retrieval_success

        generator = AnswerGenerator(
            retriever=self.mock_retriever,
            prompt_builder=self.prompt_builder,
            mock_mode=True
        )

        stream = list(generator.generate_stream(
            query="سقف مرخصی سالانه چند روز است؟",
            chat_id="test_chat_1"
        ))

        # بررسی وجود توکن‌های متنی
        token_items = [item for item in stream if item.get("type") == "token"]
        self.assertGreater(len(token_items), 0)

        # بررسی وجود مراجع در انتهای استریم
        sources_items = [item for item in stream if item.get("type") == "sources"]
        self.assertEqual(len(sources_items), 1)

        citations = sources_items[0].get("citations", [])
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["doc_id"], "hr_policy.pdf")
        self.assertEqual(citations[0]["page_number"], 4)
        self.assertEqual(citations[0]["score"], 0.92)
        self.assertIn("۲۶ روز کاری", citations[0]["snippet"])

    def test_generate_response_full(self):
        """آزمون: خروجی دیکشنری یکپارچه تابع غیر استریمی generate_response"""
        self.mock_retriever.retrieve.return_value = self.sample_retrieval_success

        generator = AnswerGenerator(
            retriever=self.mock_retriever,
            prompt_builder=self.prompt_builder,
            mock_mode=True
        )

        res = generator.generate_response(
            query="سقف مرخصی چند روزه؟",
            chat_id=10
        )

        self.assertIn("answer", res)
        self.assertIn("citations", res)
        self.assertTrue(res["has_relevant_context"])
        self.assertEqual(len(res["citations"]), 1)
        self.assertIn("۲۶ روز", res["answer"])

    def test_async_get_rag_response_stream(self):
        """آزمون: اجرای ناهمگام (AsyncGenerator) هماهنگ با قرارداد فرانت‌اند مشکات"""
        self.mock_retriever.retrieve.return_value = self.sample_retrieval_success

        generator = AnswerGenerator(
            retriever=self.mock_retriever,
            prompt_builder=self.prompt_builder,
            mock_mode=True
        )

        async def run_async_test():
            results = []
            async for chunk in generator.get_rag_response_stream(
                query="تست ناهمگام",
                chat_id=1
            ):
                results.append(chunk)
            return results

        stream_results = asyncio.run(run_async_test())

        self.assertGreater(len(stream_results), 1)
        self.assertEqual(stream_results[-1]["type"], "sources")
        self.assertGreater(len(stream_results[-1]["citations"]), 0)


if __name__ == "__main__":
    unittest.main()
