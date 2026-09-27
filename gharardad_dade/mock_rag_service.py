"""
mock_rag_service.py
----------------------------------
سرویس شبیه‌ساز (Mock) برای تست مستقل رابط کاربری (UI) توسط مشکات.
این ماژول به مشکات اجازه می‌دهد بدون نیاز به تکمیل بودن بک‌اند، تمام سناریوهای زیر را تست کند:
۱. مشاهده پیشرفت زنده پردازش اسناد (۶ مرحله تا تکمیل پردازش).
۲. قفل شدن و پیام بازدارنده چت در حین پردازش.
۳. استریم زنده کلمات و نمایش کارت مراجع و شماره صفحات در انتها.
"""

import asyncio
from typing import AsyncGenerator, Dict, Any, List, Optional
from gharardad_dade.ui_schemas import (
    STATUS_COMPLETED,
    ProgressCallback,
    Citation,
    StreamChunk
)

# مراحل استاندارد پردازش با تاخیر برای تست UI
MOCK_PIPELINE_STAGES = [
    ("در حال خواندن فایل PDF...", 0.10, 0.8),
    ("استخراج متن و پاکسازی صفحات...", 0.30, 1.0),
    ("قطعه‌بندی معنایی متن سند (چانک‌بندی)...", 0.55, 0.9),
    ("تولید بردارها با مدل هوش مصنوعی...", 0.80, 1.2),
    ("نمایه‌سازی در پایگاه داده برداری...", 0.95, 0.8),
    (STATUS_COMPLETED, 1.00, 0.5),
]


async def mock_process_document_pipeline(
    file_path: str,
    chat_id: int,
    doc_id: str,
    on_progress: Optional[ProgressCallback] = None
) -> bool:
    """
    شبیه‌ساز پردازش فایل برای تست نوار پیشرفت و وضعیت در NiceGUI.
    """
    for message, progress, delay in MOCK_PIPELINE_STAGES:
        await asyncio.sleep(delay)
        if on_progress:
            on_progress(message, progress)

    return True


async def mock_get_rag_response_stream(
    query: str,
    chat_id: int,
    filter_doc_ids: Optional[List[str]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    شبیه‌ساز استریم پاسخ مدل زبانی همراه با ارسال مراجع در انتهای پاسخ.
    """
    # متن پاسخ تستی
    mock_answer = (
        f"پاسخ استخراج‌شده برای سوال «{query}» بر اساس اسناد پردازش‌شده:\n\n"
        "بر اساس تحلیل محتوای اسناد، راهکار پیشنهادی استفاده از مدل‌های آفلاین سبک‌وزن است "
        "تا بدون نیاز به اینترنت و با کمترین مصرف حافظه پاسخ‌های دقیق تولید شود."
    )

    # ۱. شبیه‌سازی ارسال کلمه به کلمه (توکن‌ها)
    for word in mock_answer.split(" "):
        yield StreamChunk(type="token", content=word + " ").to_dict()
        await asyncio.sleep(0.05)

    # ۲. ارسال مراجع و استنادها (Citations) در پایان پاسخ
    mock_citations = [
        Citation(
            doc_id="doc_0001",
            filename="گزارش_پردازش_زبان_طبیعی.pdf",
            page_number=2,
            score=0.94,
            snippet="نویسنده استدلال می‌کند که مدل‌های سبک‌وزن برای اجرای آفلاین مناسب‌ترند."
        ),
        Citation(
            doc_id="doc_0001",
            filename="گزارش_پردازش_زبان_طبیعی.pdf",
            page_number=3,
            score=0.82,
            snippet="کاهش اندازه مدل تاثیر منفی محسوسی بر دقت پاسخ‌ها در زبان فارسی نداشت."
        ),
    ]

    yield StreamChunk(type="sources", citations=mock_citations).to_dict()
