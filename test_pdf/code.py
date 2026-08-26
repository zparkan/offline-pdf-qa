import pymupdf


def extract_and_chunk_pdf(
    pdf_path: str, chunk_size: int = 400, overlap: int = 50
):
    """استخراج متن از PDF و تبدیل آن به تکه‌های متنی (Chunks) همراه با متاداده."""
    doc = pymupdf.open(pdf_path)
    chunks = []

    print(f"📖 در حال پردازش فایل '{pdf_path}' با {len(doc)} صفحه...\n")

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()

        # پاک‌سازی خطوط خالی و فاصله‌های اضافه
        cleaned_text = " ".join(text.split())

        if not cleaned_text:
            continue

        # تقسیم متن صفحه به چانک‌ها با هم‌پوشانی (Sliding Window)
        start = 0
        while start < len(cleaned_text):
            end = start + chunk_size
            chunk = cleaned_text[start:end]

            chunks.append(
                {
                    "page": page_num,
                    "chunk_id": len(chunks) + 1,
                    "content": chunk,
                }
            )

            # جلو بردن پنجره با در نظر گرفتن هم‌پوشانی
            start += chunk_size - overlap

    doc.close()
    return chunks


# --- اجرای تست ---
pdf_file = "sample.pdf"  # نام فایل PDF شما

all_chunks = extract_and_chunk_pdf(pdf_file, chunk_size=300, overlap=40)

print(f"✅ تعداد کل چانک‌های تولیدشده: {len(all_chunks)}\n")

# نمایش ۳ چانک اول به عنوان نمونه
for item in all_chunks[:3]:
    print(f"--- [چانک شماره {item['chunk_id']} | صفحه {item['page']}] ---")
    print(item["content"])
    print("-" * 50)
