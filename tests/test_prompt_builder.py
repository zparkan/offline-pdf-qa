"""
tests/test_prompt_builder.py
----------------------------------
مجموعه آزمون‌های واحد (Unit Tests) برای ماژول prompt_builder.py.

موارد تحت آزمون:
  ۱. دستورالعمل سیستمی ضد توهم و الزامات استناددهی به منبع.
  ۲. سازگاری فرمت کانتکست با خروجی ریتریور (retriever.retrieve) و DocumentChunk.
  ۳. پالایش و مدیریت طول تاریخچه گفتگو (Chat History Limiting).
  ۴. ساختار پیام‌های استاندارد (Role / Messages).
  ۵. ساخت خروجی متنی ChatML برای مدل‌های زبانی سری Qwen.
  ۶. گارد اعتبارسنجی زمینه (has_sufficient_context) برای خروج سریع.
"""

import sys
import unittest
from pathlib import Path

# افزودن ریشه پروژه به مسیر پایتون برای دسترسی به ماژول‌ها
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from prompt_builder import PromptBuilder, DEFAULT_SYSTEM_INSTRUCTION, FALLBACK_NO_CONTEXT_MESSAGE
from schemas import DocumentChunk


class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder(max_history_turns=2)

        # نمونه دیتای خروجی واقعی از ماژول retriever
        self.mock_retriever_output = {
            "query_text": "ساعت کاری شرکت چقدر است؟",
            "top_chunks": [
                {
                    "chunk_id": "doc1_c001",
                    "text": "ساعت کاری شرکت از ساعت ۸ صبح تا ۱۶:۳۰ عصر می‌باشد.",
                    "doc_id": "rules.pdf",
                    "page_number": 2,
                    "score": 0.032,
                    "similarity": 0.88,
                    "dense_rank": 1,
                    "bm25_rank": 1
                },
                {
                    "chunk_id": "doc1_c002",
                    "text": "روزهای پنج‌شنبه و جمعه شرکت تعطیل رسمی است.",
                    "doc_id": "rules.pdf",
                    "page_number": 3,
                    "score": 0.016,
                    "similarity": 0.75,
                    "dense_rank": 2,
                    "bm25_rank": None
                }
            ],
            "has_relevant_context": True,
            "max_similarity": 0.88,
            "elapsed_ms": 12.5
        }

    def test_system_instruction_anti_hallucination(self):
        """آزمون: اطمینان از وجود اصول ضد توهم و الزام ذکر شماره صفحه در دستور سیستمی"""
        instruction = self.builder.system_instruction
        self.assertIn("فقط و فقط", instruction)
        self.assertIn("شماره صفحه", instruction)
        self.assertIn("پاسخی برای این پرسش در اسناد بارگذاری‌شده شما یافت نشد", instruction)

    def test_format_context_from_retriever_dict(self):
        """آزمون: تبدیل خروجی دیکشنری بازیاب به متن مرتب همراه با منبع و صفحه"""
        context_str = self.builder.format_context(self.mock_retriever_output)

        self.assertIn("[منبع شماره 1 | سند: rules.pdf | شماره صفحه: 2]", context_str)
        self.assertIn("ساعت کاری شرکت از ساعت ۸ صبح تا ۱۶:۳۰ عصر می‌باشد.", context_str)
        self.assertIn("[منبع شماره 2 | سند: rules.pdf | شماره صفحه: 3]", context_str)
        self.assertIn("روزهای پنج‌شنبه و جمعه شرکت تعطیل رسمی است.", context_str)
        self.assertIn("---", context_str)

    def test_format_context_from_document_chunks(self):
        """آزمون: تبدیل لیستی از اشیای کلاس DocumentChunk به متن زمینه"""
        chunks = [
            DocumentChunk(
                chunk_id="hr_c001",
                doc_id="hr_policy.pdf",
                chunk_index=0,
                text="مرخصی استحقاقی سالانه ۲۶ روز کاری است.",
                page_number=5,
                char_start=10,
                char_end=50
            )
        ]
        context_str = self.builder.format_context(chunks)

        self.assertIn("[منبع شماره 1 | سند: hr_policy.pdf | شماره صفحه: 5]", context_str)
        self.assertIn("مرخصی استحقاقی سالانه ۲۶ روز کاری است.", context_str)

    def test_chat_history_pruning(self):
        """آزمون: محدود کردن طول تاریخچه چت به اندازه مجاز max_history_turns (۲ ترن = ۴ پیام)"""
        long_history = [
            {"role": "user", "content": "سوال ۱"},
            {"role": "assistant", "content": "جواب ۱"},
            {"role": "user", "content": "سوال ۲"},
            {"role": "assistant", "content": "جواب ۲"},
            {"role": "user", "content": "سوال ۳"},
            {"role": "assistant", "content": "جواب ۳"},
        ]
        pruned = self.builder.format_history(long_history)

        # حداکثر ۴ پیام آخر باید بمانند
        self.assertEqual(len(pruned), 4)
        self.assertEqual(pruned[0]["content"], "سوال ۲")
        self.assertEqual(pruned[-1]["content"], "جواب ۳")

    def test_build_messages_structure(self):
        """آزمون: بررسی صحت ساختار لیست پیام‌ها برای ورودی به llama-cpp"""
        history = [
            {"role": "user", "content": "سلام"},
            {"role": "assistant", "content": "درود، در خدمتم."}
        ]
        messages = self.builder.build_messages(
            query="ساعت کار چنده؟",
            retriever_result_or_chunks=self.mock_retriever_output,
            chat_history=history
        )

        # ساختار باید به ترتیب: system -> history(user, assistant) -> user_prompt باشد
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "سلام")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[3]["role"], "user")

        # بررسی وجود زمینه و سوال در پیام آخر کاربر
        last_user_msg = messages[3]["content"]
        self.assertIn("### زمینه‌ها و اسناد مرجع:", last_user_msg)
        self.assertIn("ساعت کاری شرکت", last_user_msg)
        self.assertIn("ساعت کار چنده؟", last_user_msg)

    def test_build_chatml_prompt(self):
        """آزمون: تولید قالب متنی استاندارد ChatML"""
        messages = [
            {"role": "system", "content": "تو یک دستیار هستی."},
            {"role": "user", "content": "سلام، چطوری؟"}
        ]
        chatml = self.builder.build_chatml_prompt(messages)

        self.assertIn("<|im_start|>system\nتو یک دستیار هستی.<|im_end|>", chatml)
        self.assertIn("<|im_start|>user\nسلام، چطوری؟<|im_end|>", chatml)
        self.assertTrue(chatml.endswith("<|im_start|>assistant\n"))

    def test_has_sufficient_context_and_fast_exit(self):
        """آزمون: اعتبارسنجی گارد خروج سریع بر اساس پرچم has_relevant_context"""
        # حالت زمینه معتبر
        self.assertTrue(self.builder.has_sufficient_context(self.mock_retriever_output))

        # حالت عدم وجود زمینه معتبر (ارتباط پایین)
        irrelevant_output = {
            "query_text": "پایتخت برزیل کجاست؟",
            "top_chunks": [],
            "has_relevant_context": False
        }
        self.assertFalse(self.builder.has_sufficient_context(irrelevant_output))

        # حالت لیست خالی
        self.assertFalse(self.builder.has_sufficient_context([]))


if __name__ == "__main__":
    unittest.main()
