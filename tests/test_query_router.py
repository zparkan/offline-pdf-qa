"""
tests/test_query_router.py
-------------------------------------------------
آزمون‌های جامع مسیریاب هوشمند و استراتژی‌های تحلیل کلان اسناد (گام دوم حل مشکل ۸):
  ۱. آزمون دسته‌بندی قصد پرسش (Intent Classification)
  ۲. آزمون استخراج شماره اسناد هدف (Target Doc Resolution)
  ۳. آزمون استخراج واژه هدف در سوالات شمارشی (Count Target Extraction)
  ۴. آزمون شمارش قطعی و ریاضی واژگان (Deterministic Keyword Count)
  ۵. آزمون خلاصه‌سازی جامع سند با چیدمان بر اساس شماره صفحه (Summarization Context)
  ۶. آزمون مقایسه تطبیقی دو سند (Comparison Context)
  ۷. آزمون یکپارچگی استریم پاسخ‌ساز (AnswerGenerator E2E Routing)
"""

import sys
import unittest
from pathlib import Path

# تنظیم کدگذاری خروجی
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from query_router import (
    QueryIntent,
    analyze_query,
    extract_count_keyword,
    extract_target_doc_indices,
    execute_keyword_count,
    build_document_context_for_summary,
    build_comparison_context,
)
from answer_generator import AnswerGenerator
from schemas import DocumentChunk


class TestQueryRouter(unittest.TestCase):
    """
    مجموعه آزمون‌های اعتبارسنجی سیستم مسیریابی و پردازش کلان اسناد
    """

    def test_1_intent_classification(self):
        """آزمون تشخیص دقیق قصد پرسش کاربر در ۴ دسته اصلی."""
        # شمارشی و تجمعی
        self.assertEqual(
            analyze_query("تو کل متن این اسناد چند بار کلمه لپ‌تاپ تکرار شده؟").intent,
            QueryIntent.AGGREGATION_COUNT
        )
        self.assertEqual(
            analyze_query("چند بار عبارت هوش مصنوعی در متن آمده؟").intent,
            QueryIntent.AGGREGATION_COUNT
        )
        self.assertEqual(
            analyze_query("تعداد دفعات تکرار کلمه تهران چقدر است؟").intent,
            QueryIntent.AGGREGATION_COUNT
        )

        # خلاصه‌سازی
        self.assertEqual(
            analyze_query("سند شماره ۴ رو خلاصه کن").intent,
            QueryIntent.SUMMARIZATION
        )
        self.assertEqual(
            analyze_query("چکیده این فایل را بنویس").intent,
            QueryIntent.SUMMARIZATION
        )
        self.assertEqual(
            analyze_query("میشه یک مرور کلی از کل اسناد ارائه بدی؟").intent,
            QueryIntent.SUMMARIZATION
        )

        # مقایسه‌ای
        self.assertEqual(
            analyze_query("پاسخنامه شماره ۵ و ۶ رو با هم مقایسه کن").intent,
            QueryIntent.COMPARISON
        )
        self.assertEqual(
            analyze_query("تفاوت سند اول و دوم چیست؟").intent,
            QueryIntent.COMPARISON
        )
        self.assertEqual(
            analyze_query("این دو سند چه فرقی با هم دارند؟").intent,
            QueryIntent.COMPARISON
        )

        # فکتوال نقطه‌ای
        self.assertEqual(
            analyze_query("ساعت کاری شرکت چقدر است؟").intent,
            QueryIntent.FACTUAL_POINT
        )
        self.assertEqual(
            analyze_query("سقف بیمه تکمیلی پرسنل چقدر است؟").intent,
            QueryIntent.FACTUAL_POINT
        )

    def test_2_target_doc_and_keyword_extraction(self):
        """آزمون استخراج واژگان هدف و شماره اسناد در کوئری‌های پیچیده."""
        # استخراج واژه هدف
        self.assertEqual(
            extract_count_keyword("تو کل متن این اسناد چند بار کلمه لپ‌تاپ تکرار شده؟"),
            "لپ‌تاپ"
        )
        self.assertEqual(
            extract_count_keyword("چند بار «شبکه عصبی» ذکر شده؟"),
            "شبکه عصبی"
        )
        self.assertEqual(
            extract_count_keyword("کلمه بودجه چند بار آمده؟"),
            "بودجه"
        )

        # استخراج شماره اسناد هدف
        self.assertEqual(extract_target_doc_indices("سند شماره ۴ رو خلاصه کن"), [4])
        self.assertEqual(extract_target_doc_indices("پاسخنامه شماره ۵ و ۶ رو با هم مقایسه کن"), [5, 6])
        self.assertEqual(extract_target_doc_indices("سند اول و دوم رو با هم مقایسه کن"), [1, 2])

    def test_3_deterministic_keyword_count_from_chunks(self):
        """آزمون شمارش دقیق واژگان با داده‌های ساختگی چانک‌ها."""
        chunks_data = {
            "documents": [
                "در این شرکت از لپ‌تاپ برای برنامه‌نویسی استفاده می‌شود.",
                "قیمت هر لپ‌تاپ اداری حدود ۳۰ میلیون تومان است. لپ‌تاپ گیمینگ گران‌تر است.",
                "قوانین حضور و غیاب پرسنل در سازمان."
            ],
            "metadatas": [
                {"filename": "doc1.pdf", "doc_id": "doc1", "page_number": 1},
                {"filename": "doc1.pdf", "doc_id": "doc1", "page_number": 2},
                {"filename": "doc2.pdf", "doc_id": "doc2", "page_number": 1}
            ]
        }

        files = [{"id": 1, "filename": "doc1.pdf"}, {"id": 2, "filename": "doc2.pdf"}]
        res = execute_keyword_count(
            chat_id="test_chat",
            keyword="لپ‌تاپ",
            files=files,
            chunks_data=chunks_data
        )

        # باید دقیقاً ۳ بار در doc1 (۱ بار در ص ۱ و ۲ بار در ص ۲) پیدا شده باشد
        self.assertEqual(res["total_count"], 3)
        self.assertEqual(len(res["by_file"]), 1)
        self.assertEqual(res["by_file"][0]["filename"], "doc1.pdf")
        self.assertEqual(res["by_file"][0]["pages"][1], 1)
        self.assertEqual(res["by_file"][0]["pages"][2], 2)
        self.assertGreater(len(res["citations"]), 0)

    def test_4_full_document_context_assembly(self):
        """آزمون گردآوری کانتکست کامل اسناد برای خلاصه‌سازی و مقایسه."""
        chunks_data = {
            "documents": [
                "بخش اول: معرفی سازمان و تاریخچه تاسیس شرکت.",
                "بخش دوم: ساختار مالی و صورت‌های سود و زیان سال مالی.",
                "گزارش عملکرد واحد بازاریابی و فروش در شش‌ماهه اول."
            ],
            "metadatas": [
                {"filename": "report_a.pdf", "doc_id": "docA", "page_number": 1, "chunk_index": 0},
                {"filename": "report_a.pdf", "doc_id": "docA", "page_number": 2, "chunk_index": 1},
                {"filename": "report_b.pdf", "doc_id": "docB", "page_number": 1, "chunk_index": 0}
            ]
        }

        # تست کانتکست خلاصه
        file_a = {"id": 1, "filename": "report_a.pdf", "doc_id": "docA"}
        sum_ctx = build_document_context_for_summary(
            chat_id="test_chat",
            target_files=[file_a],
            chunks_data=chunks_data
        )
        self.assertIn("report_a.pdf", sum_ctx["formatted_context"])
        self.assertIn("شماره صفحه: 1", sum_ctx["formatted_context"])
        self.assertIn("شماره صفحه: 2", sum_ctx["formatted_context"])
        self.assertEqual(len(sum_ctx["citations"]), 2)

        # تست کانتکست مقایسه
        file_b = {"id": 2, "filename": "report_b.pdf", "doc_id": "docB"}
        comp_ctx = build_comparison_context(
            chat_id="test_chat",
            file_a=file_a,
            file_b=file_b,
            chunks_data=chunks_data
        )
        self.assertIn("مستندات سند اول", comp_ctx["formatted_context"])
        self.assertIn("مستندات سند دوم", comp_ctx["formatted_context"])
        self.assertIn("report_a.pdf", comp_ctx["formatted_context"])
        self.assertIn("report_b.pdf", comp_ctx["formatted_context"])

    def test_5_answer_generator_e2e_routing(self):
        """آزمون جریان استریم کامل در AnswerGenerator با هدایت هوشمند کوئری‌ها."""
        ans_gen = AnswerGenerator(mock_mode=True)

        # تست کوئری بدون سند در چت خالی
        packets = list(ans_gen.generate_stream(
            query="ساعت کاری چقدر است؟",
            chat_id="empty_chat_for_test"
        ))
        tokens = [p["content"] for p in packets if p.get("type") == "token"]
        full_text = "".join(tokens)
        self.assertIn("اسناد بارگذاری‌شده", full_text)


if __name__ == "__main__":
    unittest.main()
