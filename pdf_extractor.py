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


def extract_pdf(pdf_path: str | Path, source_type: str = "real") -> ExtractedDocument:
    """
    استخراج متن یک فایل PDF به تفکیک صفحه.

    Args:
        pdf_path: مسیر فایل PDF ورودی.
        source_type: "real" برای PDF واقعی، "synthetic" فقط برای سازگاری با F-05.

    Returns:
        ExtractedDocument شامل doc_id, filename, source_type و لیست pages.

    Raises:
        FileNotFoundError: اگر فایل وجود نداشته باشد.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"فایل پیدا نشد: {pdf_path}")

    doc_id = _generate_doc_id(pdf_path)
    pages: list[ExtractedPage] = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            # نکته‌ی باز: اگر صفحه لایه‌ی متنی نداشته باشد (اسکن‌شده)،
            # get_text() رشته‌ی خالی برمی‌گرداند. طبق تصمیم فعلی این صفحه
            # از لیست حذف نمی‌شود، ولی اینکه نیاز به OCR جداگانه دارد یا نه
            # هنوز با تیم مشخص نشده.
            #
            # sort=True: طبق تست F-03 روی shimi.pdf (جزوه‌ی چندستونی/جدولی)،
            # این گزینه ترتیب خواندن (reading order) را به‌شکل محسوسی بهتر
            # می‌کند -- تعداد شکستگی‌های \n وسط کلمه در یک صفحه‌ی فهرست‌مانند
            # از ۱۲۲ به ۲۸ مورد کاهش یافت. این مشکل را کامل حل نمی‌کند
            # (اسناد چندستونی هنوز محدودیت دارند) ولی هزینه‌ای ندارد و روی
            # اسناد پاراگرافی ساده (مثل sample_persian.pdf) تاثیر منفی
            # نداشت.
            raw_text = page.get_text("text", sort=True)
            pages.append(
                ExtractedPage(
                    page_number=page_index + 1,  # PyMuPDF از صفر شروع می‌کند، ما از ۱
                    raw_text=raw_text,
                )
            )

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
