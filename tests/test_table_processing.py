"""
tests/test_table_processing.py
-------------------------------------------------
آزمون‌های جامع استخراج ساختاریافته جداول و قطعه‌بندی هوشمند اتمیک (گام اول حل مشکل ۸):
  ۱. آزمون استخراج متون عادی (بدون جدول) و حفظ سازگاری کامل
  ۲. آزمون استخراج جدول به فرمت Markdown و ادغام بدون همپوشانی با متون صفحه در pdf_extractor.py
  ۳. آزمون برش هوشمند جداول و تکرار هدر در قطعات جداگانه در chunk_manager.py
  ۴. آزمون حفظ اتمیک جدول ۱۲ سطری در یک چانک مجزا به منظور جلوگیری از خطای شمارش و گم شدن سطرها
"""

import sys
import unittest
from pathlib import Path
import pymupdf as fitz

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
from pdf_extractor import _extract_page_text_and_tables, extract_pdf
from chunk_manager import _split_table_block, chunk_page_text, chunk_document
from schemas import ExtractedDocument, PreprocessedDocument, PreprocessedPage


class TestTableProcessing(unittest.TestCase):
    """
    مجموعه آزمون‌های اعتبارسنجی استخراج و قطعه‌بندی جداول در اسناد PDF
    """

    def test_1_plain_page_extraction_backward_compatibility(self):
        """آزمون استخراج صفحه بدون جدول: متن ساده بدون تغییر استخراج می‌شود."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text([50, 50], "This is a simple paragraph.")
        page.insert_text([50, 100], "This is the second paragraph.")

        extracted = _extract_page_text_and_tables(page)
        self.assertIn("This is a simple paragraph.", extracted)
        self.assertIn("This is the second paragraph.", extracted)
        self.assertNotIn("|", extracted)
        doc.close()

    def test_2_table_extraction_to_markdown_with_surrounding_text(self):
        """آزمون استخراج هوشمند صفحه حاوی جدول: تبدیل جدول به مارک‌داون و حفظ چینش عمودی."""
        doc = fitz.open()
        page = doc.new_page()

        # متن قبل از جدول
        page.insert_text([50, 40], "Sales Report Header")

        # رسم یک جدول ۲ ستونه و ۴ سطره
        x0, y0, x1, y1 = 50, 60, 300, 180
        page.draw_rect([x0, y0, x1, y1])
        row_h = (y1 - y0) / 4
        page.draw_line([180, y0], [180, y1])
        for i in range(1, 4):
            page.draw_line([x0, y0 + i * row_h], [x1, y0 + i * row_h])

        page.insert_text([60, y0 + row_h * 0.7], "Product")
        page.insert_text([200, y0 + row_h * 0.7], "Qty")
        page.insert_text([60, y0 + 1.7 * row_h], "Laptop")
        page.insert_text([200, y0 + 1.7 * row_h], "15")
        page.insert_text([60, y0 + 2.7 * row_h], "Keyboard")
        page.insert_text([200, y0 + 2.7 * row_h], "30")
        page.insert_text([60, y0 + 3.7 * row_h], "Mouse")
        page.insert_text([200, y0 + 3.7 * row_h], "50")

        # متن بعد از جدول
        page.insert_text([50, 220], "Footer Notes and Conclusion.")

        extracted = _extract_page_text_and_tables(page)

        # بررسی ساختار جدول در خروجی
        self.assertIn("Sales Report Header", extracted)
        self.assertIn("|", extracted)
        self.assertIn("Product", extracted)
        self.assertIn("Qty", extracted)
        self.assertIn("Laptop", extracted)
        self.assertIn("Keyboard", extracted)
        self.assertIn("Mouse", extracted)
        self.assertIn("Footer Notes and Conclusion.", extracted)

        # موقعیت جدول باید بین عنوان و متن پایانی باشد
        pos_title = extracted.find("Sales Report Header")
        pos_table = extracted.find("|")
        pos_footer = extracted.find("Footer Notes and Conclusion.")
        self.assertTrue(pos_title < pos_table < pos_footer, "چینش عمودی عناصر در صفحه به هم ریخته است!")
        doc.close()

    def test_3_split_table_block_preserves_headers(self):
        """آزمون تقسیم جدول بزرگ با تکرار هدرها در تمام قطعات."""
        header = "| ستون الف | ستون ب |\n| :--- | :--- |\n"
        rows = "\n".join([f"| داده شماره {i} | مقدار {i * 10} |" for i in range(1, 20)])
        table_text = header + rows

        # جدول با محدودیت اندازه کوچک به چند بخش تقسیم شود
        pieces = _split_table_block(table_text, max_size=250)
        self.assertGreater(len(pieces), 1)

        for p in pieces:
            # هر قطعه باید حتماً سطر عنوان و جداکننده را در ابتدا داشته باشد
            self.assertTrue(p.startswith("| ستون الف | ستون ب |"))
            self.assertIn("| :--- | :--- |", p)

    def test_4_atomic_12_row_table_chunking(self):
        """آزمون قطعه‌بندی اتمیک جدول ۱۲ سطری: کل ۱۲ سطر باید در یک چانک یکپارچه باقی بمانند."""
        header = "| ردیف | نام لپ‌تاپ | تعداد |\n| --- | --- | --- |\n"
        rows = "\n".join([f"| {i} | لپ‌تاپ مدل شماره {i} | {i * 3} |" for i in range(1, 13)])
        full_page_text = f"مقدمه صفحه اول.\n\n{header}{rows}\n\nنتیجه‌گیری گزارش."

        chunks = chunk_page_text(
            doc_id="doc_table_test",
            page_number=1,
            text=full_page_text,
            chunk_size=500,
            table_max_size=1200
        )

        # باید حداقل ۲ چانک تولید شده باشد و یکی از آن‌ها دقیقاً حاوی کل جدول باشد
        table_chunks = [c for c in chunks if "| --- | --- | --- |" in c.text]
        self.assertEqual(len(table_chunks), 1, "جدول ۱۲ سطری نباید قطعه‌قطعه می‌شد!")

        t_chunk = table_chunks[0]
        # بررسی حضور تمام ۱۲ سطر در همان یک چانک
        for i in range(1, 13):
            self.assertIn(f"لپ‌تاپ مدل شماره {i}", t_chunk.text, f"سطر {i} از جدول مفقود شده است!")

    def test_5_chunk_document_integration(self):
        """آزمون قطعه‌بندی سطح سند (chunk_document) با صفحات حاوی جدول."""
        header = "| شناسه | کالا |\n| --- | --- |\n"
        rows = "\n".join([f"| {i} | مورد {i} |" for i in range(1, 6)])
        table_text = header + rows

        prep_doc = PreprocessedDocument(
            doc_id="doc_integration_table",
            filename="test_table.pdf",
            source_type="real",
            pages=[
                PreprocessedPage(
                    page_number=1,
                    raw_text="",
                    clean_text=f"صفحه حاوی جدول:\n\n{table_text}"
                )
            ]
        )

        chunks = chunk_document(prep_doc)
        self.assertGreater(len(chunks), 0)
        table_chunk = [c for c in chunks if "| شناسه | کالا |" in c.text][0]
        for i in range(1, 6):
            self.assertIn(f"مورد {i}", table_chunk.text)


if __name__ == "__main__":
    unittest.main()
