"""
chunk_manager.py — ماژول قطعه‌بندی هوشمند متن اسناد (F-04)
---------------------------------------------------------------------
مسئولیت:
  ۱. دریافت متن پالایش‌شده صفحات سند (خروجی F-03 / text_preprocessor.py).
  ۲. برش هوشمند متن با حفظ تمامیت جملات و جلوگیری از شکستن کلمات.
  ۳. اعمال همپوشانی (Overlap) برای حفظ پیوستگی معنایی در مرزهای قطعات.
  ۴. رهگیری دقیق شماره صفحه (page_number) و بازه کاراکتری (char_start, char_end) برای ارجاع دقیق (Citation).
  ۵. تولید لیستی از اشیاء رسمی DocumentChunk منطبق با schemas.py.
  ۶. ذخیره‌سازی داده‌های ساخت‌یافته در قالب JSON در پوشه اختصاصی گفتگوی مربوطه.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable

# تنظیم خروجی ترمینال ویندوز روی UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from schemas import DocumentChunk, PreprocessedDocument


# الگوهای جداکننده جملات در زبان فارسی و انگلیسی برای برش طبیعی متن
_SENTENCE_SPLIT_REGEX = re.compile(r"([.!?؟؛\n]+)")

# الگوی تشخیص بلوک جداول مارک‌داون (حاوی سطر عنوان، خط جداکننده و سطرهای داده)
_TABLE_BLOCK_REGEX = re.compile(
    r"((?:^[ \t]*\|.+?\|[ \t]*\r?\n)+(?:^[ \t]*\|[-: |]+?\|[ \t]*\r?\n)(?:^[ \t]*\|.+?\|[ \t]*(?:\r?\n|$))+)",
    re.MULTILINE
)


def _split_table_block(table_text: str, max_size: int = 1200) -> List[str]:
    """
    برش هوشمند جداول بسیار بزرگ با حفظ و تکرار هدر جدول در تمام قطعات.
    """
    lines = [l for l in table_text.strip().splitlines() if l.strip()]
    if len(lines) <= 2 or len(table_text.strip()) <= max_size:
        return [table_text.strip()]

    header_block = lines[0] + "\n" + lines[1] + "\n"
    data_rows = lines[2:]
    chunks = []
    current_rows = []
    current_len = len(header_block)

    for row in data_rows:
        row_len = len(row) + 1
        if current_len + row_len > max_size and current_rows:
            chunks.append(header_block + "\n".join(current_rows))
            current_rows = []
            current_len = len(header_block)
        current_rows.append(row)
        current_len += row_len

    if current_rows:
        chunks.append(header_block + "\n".join(current_rows))

    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """
    تقسیم متن به جملات مجزا بدون از بین بردن علائم نگارشی.
    """
    if not text:
        return []
    parts = _SENTENCE_SPLIT_REGEX.split(text)
    sentences = []
    # ترکیب بخش متنی با علامت نگارشی پس از آن
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i] + parts[i + 1]
        if sentence.strip():
            sentences.append(sentence)
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1])
    return sentences if sentences else [text]


def chunk_page_text(
    doc_id: str,
    page_number: int,
    text: str,
    start_index: int = 0,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    filename: str = "",
    table_max_size: Optional[int] = None,
) -> List[DocumentChunk]:
    """
    قطعه‌بندی هوشمند متن یک صفحه بر اساس اندازه و همپوشانی، با محافظت ویژه از ساختار جداول.

    استراتژی:
      - شناسایی بلوک‌های جدولی Markdown و حفظ آن‌ها به صورت اتمیک تا سقف TABLE_CHUNK_MAX_SIZE.
      - در صورت بزرگتر بودن جدول از سقف مجاز، سطرها تقسیم شده و عنوان ستون‌ها در بالای تمام قطعات تکرار می‌شود.
      - متن‌های عادی بر اساس مرز جملات و با همپوشانی (Overlap) قطعه‌بندی می‌شوند.
    """
    target_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    target_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    max_table = table_max_size if table_max_size is not None else getattr(config, "TABLE_CHUNK_MAX_SIZE", 1200)

    if not text or not text.strip():
        return []

    chunks: List[DocumentChunk] = []
    chunk_counter = start_index
    search_cursor = 0

    def create_chunk_object(chunk_text: str) -> DocumentChunk:
        nonlocal chunk_counter, search_cursor
        clean_chunk = chunk_text.strip()

        # محاسبه موقعیت چانک در متن اصلی صفحه
        found_pos = text.find(clean_chunk[:min(30, len(clean_chunk))], search_cursor)
        if found_pos != -1:
            c_start = found_pos
            c_end = found_pos + len(clean_chunk)
            search_cursor = max(search_cursor, c_start + 1)
        else:
            c_start = 0
            c_end = len(clean_chunk)

        c_id = f"{doc_id}_c{chunk_counter:03d}"
        item = DocumentChunk(
            chunk_id=c_id,
            doc_id=doc_id,
            chunk_index=chunk_counter,
            text=clean_chunk,
            page_number=page_number,
            char_start=c_start,
            char_end=c_end,
            filename=filename,
        )
        chunk_counter += 1
        return item

    def chunk_plain_text(plain_text: str):
        sentences = _split_into_sentences(plain_text)
        current_sentences: List[str] = []
        current_length = 0

        for sentence in sentences:
            s_len = len(sentence)

            # اگر یک جمله به تنهایی از سقف چانک بزرگتر بود
            if s_len > target_size:
                if current_sentences:
                    chunk_str = "".join(current_sentences)
                    chunks.append(create_chunk_object(chunk_str))
                    current_sentences = []
                    current_length = 0

                words = sentence.split(" ")
                temp_words: List[str] = []
                temp_len = 0
                for w in words:
                    if temp_len + len(w) + 1 > target_size and temp_words:
                        chunks.append(create_chunk_object(" ".join(temp_words)))
                        overlap_words = temp_words[-max(1, int(len(temp_words) * 0.2)):]
                        temp_words = list(overlap_words)
                        temp_len = sum(len(x) + 1 for x in temp_words)
                    temp_words.append(w)
                    temp_len += len(w) + 1
                if temp_words:
                    chunks.append(create_chunk_object(" ".join(temp_words)))
                continue

            # بررسی سرریز اندازه چانک
            if current_length + s_len > target_size and current_sentences:
                chunk_str = "".join(current_sentences)
                chunks.append(create_chunk_object(chunk_str))

                overlap_sentences: List[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) <= target_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break

                current_sentences = overlap_sentences
                current_length = overlap_len

            current_sentences.append(sentence)
            current_length += s_len

        if current_sentences:
            chunk_str = "".join(current_sentences)
            if chunk_str.strip():
                chunks.append(create_chunk_object(chunk_str))

    # شناسایی و تفکیک بخش‌های جدولی در متن صفحه
    table_matches = list(_TABLE_BLOCK_REGEX.finditer(text))
    if not table_matches:
        chunk_plain_text(text)
        return chunks

    last_idx = 0
    for match in table_matches:
        start, end = match.span()
        if start > last_idx:
            pre_text = text[last_idx:start].strip()
            if pre_text:
                chunk_plain_text(pre_text)

        table_raw = match.group(0).strip()
        table_parts = _split_table_block(table_raw, max_size=max_table)
        for part in table_parts:
            chunks.append(create_chunk_object(part))

        last_idx = end

    if last_idx < len(text):
        post_text = text[last_idx:].strip()
        if post_text:
            chunk_plain_text(post_text)

    return chunks


def chunk_document(
    doc: Union[PreprocessedDocument, Dict[str, Any]],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    table_max_size: Optional[int] = None,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> List[DocumentChunk]:
    """
    قطعه‌بندی کل صفحات یک سند و برگرداندن یک لیست مسطح (Flat List) از DocumentChunk.

    ورودی:
      - doc: نمونه PreprocessedDocument یا دیکشنری با کلیدهای doc_id و pages
      - chunk_size: اندازه سفارشی چانک به کاراکتر (در صورت None از کانفیگ خوانده می‌شود)
      - chunk_overlap: اندازه سفارشی همپوشانی به کاراکتر (در صورت None از کانفیگ خوانده می‌شود)
      - table_max_size: سقف اندازه قطعات حاوی جداول (در صورت None از کانفیگ خوانده می‌شود)
      - on_progress: کال‌بک گزارش درصد پیشرفت به رابط کاربری
    """
    # اعتبارسنجی مقادیر چانک در برابر مدل امبدینگ فعال
    c_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    c_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    t_max = table_max_size if table_max_size is not None else getattr(config, "TABLE_CHUNK_MAX_SIZE", 1200)
    config.validate_chunk_config(chunk_size=c_size)

    if isinstance(doc, PreprocessedDocument):
        doc_dict = doc.model_dump()
    else:
        doc_dict = doc

    doc_id = doc_dict.get("doc_id", "doc_unknown")
    doc_filename = doc_dict.get("filename", "")
    pages = doc_dict.get("pages", [])
    total_pages = len(pages)

    if on_progress:
        on_progress("شروع قطعه‌بندی معنایی سند...", 0.60)

    all_chunks: List[DocumentChunk] = []
    chunk_global_idx = 0

    for idx, page in enumerate(pages):
        page_num = page.get("page_number", idx + 1)
        # اولویت با متن تمیزشده clean_text است؛ اگر نبود از raw_text استفاده می‌شود
        text_content = page.get("clean_text") or page.get("raw_text", "")

        page_chunks = chunk_page_text(
            doc_id=doc_id,
            page_number=page_num,
            text=text_content,
            start_index=chunk_global_idx,
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            filename=doc_filename,
            table_max_size=t_max,
        )

        all_chunks.extend(page_chunks)
        chunk_global_idx += len(page_chunks)

        if on_progress and total_pages > 0:
            pct = 0.60 + (0.15 * ((idx + 1) / total_pages))
            on_progress(f"قطعه‌بندی صفحه {page_num} از {total_pages}...", pct)

    if on_progress:
        on_progress(f"قطعه‌بندی تکمیل شد ({len(all_chunks)} چانک تولید گردید).", 0.75)

    return all_chunks


def save_chunks_to_json(chunks: List[DocumentChunk], output_path: Union[str, Path]) -> Path:
    """
    ذخیره‌سازی چانک‌های تولیدشده در قالب یک فایل JSON در مسیر مشخص‌شده (پوشه چت).
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    data = [c.to_dict() for c in chunks]
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return out_p


def load_chunks_from_json(json_path: Union[str, Path]) -> List[DocumentChunk]:
    """
    خواندن چانک‌ها از فایل JSON و تبدیل مجدد به اشیاء DocumentChunk.
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"فایل چانک‌ها پیدا نشد: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [DocumentChunk.from_dict(item) for item in data]
