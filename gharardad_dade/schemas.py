"""
schemas.py
----------------------------------
تعریف رسمیِ ساختارهای داده‌ی مشترک بین ماژول‌های تیم (Data Ingestion <-> Core RAG <-> UI/LLM).

این فایل باید در ریشه‌ی ریپازیتوری مشترک قرار بگیرد و هر سه عضو تیم آن را
import کنند (نه این‌که هرکس نسخه‌ی خودش را از ساختار داده بازتعریف کند).
با این کار، قرارداد داده‌ای دیگر فقط یک توضیح در Markdown نیست، بلکه بخشی
از خودِ کد است — اگر فیلدی تغییر کند، بلافاصله در کد بقیه‌ی اعضا هم قابل
تشخیص می‌شود (خطای import یا type)، به‌جای این‌که مشکل دیر و در زمان اجرا
کشف شود.

مالکیت فیلدها:
  - DocumentChunk: تولید و مالکیت آن با ماژول Data Ingestion (فاطمه) است.
    ماژول Core RAG (زینب) این کلاس را فقط می‌خواند، هرگز آن را ویرایط
    نمی‌کند یا در همان فایل چیزی اضافه نمی‌کند. بردار embedding و score
    باید در ساختار/فایل جداگانه‌ای نگه داشته شوند که با chunk_id به این
    کلاس متصل می‌شود.
"""

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class DocumentChunk:
    chunk_id: str          # شناسه یکتا در کل سیستم، فرمت: {doc_id}_c{شماره سه‌رقمی}
    doc_id: str             # سند مبدا این چانک
    chunk_index: int        # ترتیب چانک داخل همان سند (از صفر)
    text: str                # متن نرمال‌شده و پاکسازی‌شده (خروجی F-03 + F-04)
    page_number: int        # شماره صفحه مبدا - الزامی برای citation
    char_start: int          # موقعیت شروع در متن اصلی صفحه
    char_end: int             # موقعیت پایان در متن اصلی صفحه
    token_count: int         # تعداد تقریبی توکن، برای مدیریت context مدل embedding

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentChunk":
        return cls(**data)


@dataclass
class QAPair:
    """
    ساختار یک جفت سوال-جواب. نسخه‌ی مصنوعی آن (پاسخ placeholder) در F-05
    صرفاً برای تست ساختاری استفاده می‌شود؛ نسخه‌ی نهایی و واقعی (با پاسخ
    دستی انسانی) در F-06 ساخته می‌شود.
    """
    qa_id: str
    doc_id: str
    question: str
    question_type: str  # یکی از: "مستقیم" | "تفسیری" | "مقایسه‌ای"
    answer_reference: str
    supporting_chunk_ids: List[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "QAPair":
        return cls(**data)
