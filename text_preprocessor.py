"""
F-03 -- خط لوله‌ی پیش‌پردازش و پاکسازی متن فارسی
بخشی از ماژول Data Ingestion در پروژه‌ی سیستم پرسش‌وپاسخ آفلاین RAG
روی اسناد PDF فارسی (تیم هوشینو قم).

ورودی: خروجی JSON ماژول F-02 (ساختار سطح ۱ قرارداد داده‌ای -- pdf_extractor.py)
خروجی: همان ساختار + فیلد clean_text در هر صفحه (raw_text دست‌نخورده باقی می‌ماند)

نصب:
    pip install pydantic hazm

اجرا از خط فرمان:
    python text_preprocessor.py path/to/extracted.json
    # خروجی: path/to/extracted_clean.json

استفاده به‌عنوان ماژول:
    from text_preprocessor import preprocess_document
    result = preprocess_document(extracted_doc_dict)  # -> PreprocessedDocument
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

from hazm import Normalizer

from schemas_f03_addition import PreprocessedDocument, PreprocessedPage

# ---------------------------------------------------------------------------
# مرحله‌ی ۰: نرمال‌سازی Unicode (NFKC) -- رفع گلیف‌های Arabic Presentation Forms
# ---------------------------------------------------------------------------
# برخی فونت‌ها (دیده‌شده در shimi.pdf) حروف را به‌جای کد استاندارد فارسی/عربی
# با گلیف‌های شکل‌ویژه (ابتدایی/وسطی/پایانی/مجزا -- بازه‌ی U+FE70 تا U+FEFF)
# کدگذاری می‌کنند. این کدها خارج از بازه‌ی حروف شناخته‌شده در این ماژول هستند،
# پس بدون این مرحله، نه regex ما و نه Hazm اصلاً این حروف را «حرف» تشخیص
# نمی‌دهند. NFKC دقیقاً برای تبدیل این‌جور کاراکترهای سازگاری (compatibility)
# به شکل پایه طراحی شده است.


def normalize_unicode_forms(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


# ---------------------------------------------------------------------------
# مرحله‌ی ۱: رفع مشکل \n وسط کلمه (باگ استخراج PyMuPDF -- طبق README_F02)
# ---------------------------------------------------------------------------

# حروف الفبای فارسی + دو حرف عربی رایج (ي، ك) که ممکن است پیش از نرمال‌سازی
# Hazm هنوز در متن خام باقی مانده باشند.
_PERSIAN_LETTERS = "ءآأؤإئابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیيك"
_LETTER = f"[{_PERSIAN_LETTERS}]"

# پیشوندهای فعلی شناخته‌شده که همیشه با نیم‌فاصله به ریشه‌ی فعل می‌چسبند
_KNOWN_PREFIXES = {"می", "نمی", "بی"}
# پسوندهای پرتکرار که همیشه با نیم‌فاصله به کلمه‌ی قبل می‌چسبند
_KNOWN_SUFFIXES = {"ها", "تر", "ترین"}

_BROKEN_WORD_PATTERN = re.compile(rf"({_LETTER}+)\n({_LETTER}+)")


def fix_broken_newlines(text: str) -> str:
    """
    \\n داخل متن را که به‌احتمال زیاد از باگ استخراج PyMuPDF آمده اصلاح می‌کند.

    استراتژی (محافظه‌کارانه، طبق تصمیم مستندشده‌ی تیم):
    - اگر بخش قبل یا بعد از \\n دقیقاً یکی از پیشوندها/پسوندهای شناخته‌شده باشد
      (می، نمی، بی، ها، تر، ترین)، با نیم‌فاصله به هم وصل می‌شوند.
    - در غیر این صورت، \\n با یک فاصله‌ی معمولی جایگزین می‌شود -- چون به احتمال
      بیشتر مرز دو کلمه‌ی مستقل است (مثال: "...آمد\\nبرای..." که نباید چسبیده شود).

    این الگو ۱۰۰٪ نیست (بدون دیکشنری کامل نمی‌شود مطمئن شد)، ولی هر دو نمونه‌ی
    مستندشده در README_F02 را درست پوشش می‌دهد و ریسک چسباندن اشتباه دو کلمه‌ی
    کامل و مستقل را پایین نگه می‌دارد.
    """

    def _join(match: re.Match) -> str:
        before, after = match.group(1), match.group(2)
        if before in _KNOWN_PREFIXES or after in _KNOWN_SUFFIXES:
            return f"{before}\u200c{after}"
        return f"{before} {after}"

    return _BROKEN_WORD_PATTERN.sub(_join, text)


# ---------------------------------------------------------------------------
# مرحله‌ی ۲: نرمال‌سازی زبانی با Hazm + جبران محدودیت شناخته‌شده‌اش
# ---------------------------------------------------------------------------

_normalizer = Normalizer()

# جبران محدودیت مستندشده‌ی Hazm (طبق تست تیم): پسوند جمع «ها» وقتی مستقیم
# چسبیده باشد (بدون نیم‌فاصله) توسط Hazm اصلاح نمی‌شود. سایر الگوهای نادرتر
# (مثل ترکیبات ثابت از نوع «بیشک») عمداً پوشش داده نمی‌شوند -- محدودیت
# شناخته‌شده و مستندشده، مشابه رویکرد F-02 با مسئله‌ی OCR.
_MISSING_HALF_SPACE_HA = re.compile(rf"({_LETTER}+)ها\b")

# کلماتی که به «ها» ختم می‌شوند ولی «ها» در آن‌ها پسوند جمع نیست، بلکه جزء
# ریشه‌ی خودِ کلمه است. این لیست از بررسی واقعی sample_persian.pdf به‌دست
# آمد: بدون این استثنا، «تنها» و «رها» به‌اشتباه جدا می‌شدند.
_HA_SUFFIX_EXCLUDED_WORDS = {"تنها", "رها", "بها", "شاها", "آنچنانها"}


def _insert_half_space_before_ha(match: re.Match) -> str:
    # از یک تابع replacement استفاده می‌کنیم (نه رشته‌ی قالب r"\1\u200cها")
    # چون re.sub رشته‌ی قالب را خودش پردازش می‌کند و \u را یک escape نامعتبر
    # می‌شناسد (در پایتون ۳.۱۲ این مورد خطای re.error می‌دهد).
    whole_word = match.group(0)
    if whole_word in _HA_SUFFIX_EXCLUDED_WORDS:
        return whole_word
    return match.group(1) + "\u200c" + "ها"


def clean_persian_text(raw_text: str) -> str:
    """پایپ‌لاین کامل پاکسازی یک رشته متن خام فارسی (متن یک صفحه)."""
    text = normalize_unicode_forms(raw_text)
    text = fix_broken_newlines(text)
    text = _normalizer.normalize(text)
    text = _MISSING_HALF_SPACE_HA.sub(_insert_half_space_before_ha, text)
    return text


# ---------------------------------------------------------------------------
# مرحله‌ی ۳: اعمال روی کل سند (خروجی F-02)
# ---------------------------------------------------------------------------

def preprocess_document(doc: dict) -> PreprocessedDocument:
    """
    doc: دیکشنری با ساختار سطح ۱ قرارداد داده‌ای (خروجی F-02 -- ExtractedDocument)
    خروجی: PreprocessedDocument با فیلد clean_text در هر صفحه؛ raw_text دست‌نخورده.
    """
    pages = [
        PreprocessedPage(
            page_number=page["page_number"],
            raw_text=page["raw_text"],
            clean_text=clean_persian_text(page["raw_text"]),
        )
        for page in doc["pages"]
    ]
    return PreprocessedDocument(
        doc_id=doc["doc_id"],
        filename=doc["filename"],
        source_type=doc.get("source_type", "real"),
        pages=pages,
    )


# ---------------------------------------------------------------------------
# اجرای خط فرمان
# ---------------------------------------------------------------------------

def _main() -> None:
    if len(sys.argv) != 2:
        print("استفاده: python text_preprocessor.py path/to/extracted.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    with input_path.open(encoding="utf-8") as f:
        raw_doc = json.load(f)

    result = preprocess_document(raw_doc)

    output_path = input_path.with_name(input_path.stem + "_clean.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    print(f"ذخیره شد: {output_path}")


if __name__ == "__main__":
    _main()
