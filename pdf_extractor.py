"""
pdf_extractor.py — ماژول اصلی F-02

مسئولیت: باز کردن یک فایل PDF با PyMuPDF (بر اساس تصمیم F-01) و تولید یک
نمونه از ExtractedDocument دقیقاً مطابق ساختار سطح ۱ در data_contract.md.

نکته‌ی مهم: این ماژول هیچ پاکسازی یا نرمال‌سازی متن انجام نمی‌دهد — raw_text
دقیقاً همان چیزی است که PyMuPDF برمی‌گرداند. مشکل فاصله‌های اضافه‌ی
گاه‌به‌گاه داخل کلمات (که در F-01 دیده شد) عمداً اینجا رفع نمی‌شود؛
رفعش طبق قرارداد داده‌ای وظیفه‌ی F-03 است.
"""

import hashlib
import sys
from pathlib import Path

import pymupdf as fitz  # نام جدید کتابخانه (fitz نام مستعار قدیمی و در حال حذف است)

from schemas_f02_addition import ExtractedDocument, ExtractedPage


def _generate_doc_id(pdf_path: Path) -> str:
    """
    doc_id یکتا و پایدار بر پایه‌ی هش نام فایل + سایز فایل.
    پایدار بودن مهم است: اگر همین فایل دوباره پردازش شود، doc_id باید ثابت
    بماند تا در طول پایپ‌لاین قابل رهگیری باشد و چانک تکراری تولید نشود.
    """
    key = f"{pdf_path.name}_{pdf_path.stat().st_size}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"doc_{digest}"


from typing import Callable, Optional


def _extract_page_text_and_tables(page: fitz.Page) -> str:
    """
    استخراج هوشمند و ساختاریافته متن صفحه به همراه تحلیل ساختار جداول.

    مراحل:
      ۱. جستجوی ساختارهای جدولی با قابلیت find_tables در PyMuPDF.
      ۲. در صورت یافتن جدول: تبدیل به جدول استاندارد Markdown جهت حفظ ارتباط ستون‌ها و سطرها.
      ۳. استخراج بلوک‌های متنی خارج از محدوده هندسی جداول (جلوگیری از تکرار متن جدول).
      ۴. چینش منظم محتوا از بالا به پایین بر اساس موقعیت عمودی (y0).
      ۵. در صورت عدم وجود جدول: استفاده مستقیم و سریع از get_text("text", sort=False).
    """
    try:
        tabs = page.find_tables()
    except Exception:
        tabs = None

    if not tabs or len(tabs.tables) == 0:
        return page.get_text("text", sort=False)

    table_bboxes: list[fitz.Rect] = []
    # المان‌ها با ساختار: (y0, x0, text)
    elements: list[tuple[float, float, str]] = []

    for tab in tabs.tables:
        try:
            tab_rect = fitz.Rect(tab.bbox)
            md_content = tab.to_markdown()
            if md_content and md_content.strip():
                table_bboxes.append(tab_rect)
                elements.append((tab_rect.y0, tab_rect.x0, md_content.strip()))
        except Exception:
            continue

    if not table_bboxes:
        return page.get_text("text", sort=False)

    # استخراج بلوک‌های متنی خارج از محدوده جداول
    try:
        blocks = page.get_text("blocks")
    except Exception:
        blocks = []

    for b in blocks:
        # بلوک متن: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type == 0 متن است
        if len(b) < 5 or (len(b) >= 7 and b[6] != 0):
            continue

        b_rect = fitz.Rect(b[:4])
        b_text = b[4].strip()
        if not b_text:
            continue

        # بررسی همپوشانی با جداول
        in_table = False
        for t_rect in table_bboxes:
            intersection = b_rect & t_rect
            if intersection.is_valid and (
                intersection.get_area() > 0.35 * b_rect.get_area() or b_rect in t_rect
            ):
                in_table = True
                break

        if not in_table:
            elements.append((b_rect.y0, b_rect.x0, b_text))

    # مرتب‌سازی عناصر بر اساس موقعیت عمودی در صفحه (از بالا به پایین)
    elements.sort(key=lambda item: (item[0], item[1]))

    return "\n\n".join(item[2] for item in elements if item[2])


def extract_pdf(
    pdf_path: str | Path,
    source_type: str = "real",
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> ExtractedDocument:
    """
    استخراج متن یک فایل PDF به تفکیک صفحه با حفظ ساختار جداول.

    Args:
        pdf_path: مسیر فایل PDF ورودی.
        source_type: "real" برای PDF واقعی، "synthetic" فقط برای سازگاری با F-05.
        on_progress: کال‌بک اختیاری برای گزارش وضعیت و درصد پیشرفت به رابط کاربری.

    Returns:
        ExtractedDocument شامل doc_id, filename, source_type و لیست pages.

    Raises:
        FileNotFoundError: اگر فایل وجود نداشته باشد.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"فایل پیدا نشد: {pdf_path}")

    if on_progress:
        on_progress("در حال خواندن فایل PDF و آماده‌سازی صفحات...", 0.10)

    doc_id = _generate_doc_id(pdf_path)
    pages: list[ExtractedPage] = []

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        for page_index, page in enumerate(doc):
            # استخراج ساختاریافته متن به همراه شناسایی جداول
            raw_text = _extract_page_text_and_tables(page)
            pages.append(
                ExtractedPage(
                    page_number=page_index + 1,  # PyMuPDF از صفر شروع می‌کند، ما از ۱
                    raw_text=raw_text,
                )
            )
            if on_progress and total_pages > 0:
                pct = 0.10 + (0.20 * ((page_index + 1) / total_pages))
                on_progress(f"استخراج صفحه {page_index + 1} از {total_pages}...", pct)

    if on_progress:
        on_progress(f"استخراج {len(pages)} صفحه با موفقیت انجام شد.", 0.30)

    return ExtractedDocument(
        doc_id=doc_id,
        filename=pdf_path.name,
        source_type=source_type,
        pages=pages,
    )


def save_extracted_document(document: ExtractedDocument, output_path: str | Path) -> None:
    """
    ذخیره‌ی یک ExtractedDocument در قالب فایل JSON، دقیقاً مطابق نمونه‌ی
    سطح ۱ در data_contract.md (doc_id, filename, source_type, pages).

    Args:
        document: نمونه‌ی ExtractedDocument که extract_pdf برگردانده.
        output_path: مسیر فایل JSON مقصد؛ اگر پوشه‌های مسیر وجود نداشته
            باشند، به‌صورت خودکار ساخته می‌شوند.
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(document.model_dump(), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("استفاده: python pdf_extractor.py path/to/file.pdf")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    result = extract_pdf(input_path)

    # نام فایل خروجی از روی نام PDF ساخته می‌شود: sample.pdf -> sample.json
    output_path = input_path.with_suffix(".json")
    save_extracted_document(result, output_path)

    print(f"خروجی با موفقیت ذخیره شد در: {output_path}")
