"""
query_router.py — ماژول مسیریابی هوشمند پرسش‌ها و تحلیل قصد کاربر (Query Router & Intent Classifier)
-------------------------------------------------------------------------------------------------
مسئولیت:
  ۱. تشخیص قصد پرسش کاربر (Intent Classification):
     - FACTUAL_POINT: سوالات فکتوال و نقطه‌ای عادی (بازیابی هیبرید Top-K)
     - SUMMARIZATION: درخواست خلاصه‌سازی یک سند یا همه اسناد
     - COMPARISON: درخواست مقایسه بین دو یا چند سند
     - AGGREGATION_COUNT: درخواست شمارش یا تحلیل فراوانی یک مفهوم/کلمه در کل اسناد
  ۲. استخراج اسناد هدف (Target Document Resolution):
     - شناسایی اسناد بر اساس شماره («سند ۴»، «پاسخنامه ۵ و ۶»، «سند اول») یا نام فایل
  ۳. استخراج کلمه کلیدی در پرسش‌های شمارشی و محاسبه ریاضی قطعی (Deterministic Counting)
  ۴. گردآوری کانتکست ساختاریافته کل سند برای خلاصه‌سازی و مقایسه جامع
"""

import re
import json
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union

import chat_database as db
try:
    from gharardad_dade.ui_schemas import Citation
except ImportError:
    from dataclasses import dataclass, asdict

    @dataclass
    class Citation:
        doc_id: str
        filename: str
        page_number: int
        score: float
        snippet: str

        def to_dict(self) -> dict:
            return asdict(self)


class QueryIntent(str, Enum):
    FACTUAL_POINT = "factual_point"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    AGGREGATION_COUNT = "aggregation_count"


@dataclass
class QueryAnalysisResult:
    intent: QueryIntent
    target_keyword: Optional[str] = None
    target_doc_indices: List[int] = field(default_factory=list)  # شماره‌های ۱-مبنا (مثلاً [1, 2])
    target_filenames: List[str] = field(default_factory=list)
    is_global: bool = False  # آیا پرسش مربوط به کل اسناد است؟


# الگوهای رجکس برای تشخیص قصد پرسش
_COUNT_PATTERNS = [
    re.compile(r"چند\s*بار.*(?:تکرار|آمده|اومده|ذکر|استفاده|شده|دیده‌می‌شود)", re.IGNORECASE),
    re.compile(r"تعداد\s*دفعات", re.IGNORECASE),
    re.compile(r"چند\s*(?:مورد|تا).*کلمه", re.IGNORECASE),
    re.compile(r"فراوانی\s*(?:کلمه|واژه|عبارت|مفهوم)", re.IGNORECASE),
    re.compile(r"چند\s*بار\s*(?:کلمه|واژه|عبارت)", re.IGNORECASE),
]

_COMPARE_PATTERNS = [
    re.compile(r"مقایسه\s*(?:کن|ای|شون|کنید|نمایید)?", re.IGNORECASE),
    re.compile(r"تفاوت(?:\s*های|\s*شون|\s*بین)?", re.IGNORECASE),
    re.compile(r"شباهت(?:\s*های|\s*شون|\s*بین)?", re.IGNORECASE),
    re.compile(r"فرق(?:\s*های|\s*شون|\s*بین)?", re.IGNORECASE),
    re.compile(r"چه\s*فرقی", re.IGNORECASE),
    re.compile(r"کدوم\s*(?:بهتر|کامل‌تر|کاملتر)", re.IGNORECASE),
]

_SUMMARIZE_PATTERNS = [
    re.compile(r"خلاصه\s*(?:کن|سازی|رو|ی|اش|ش|کنید|نمایید|ای)", re.IGNORECASE),
    re.compile(r"\bچکیده\b", re.IGNORECASE),
    re.compile(r"مرور\s*کلی", re.IGNORECASE),
    re.compile(r"توضیح\s*کلی", re.IGNORECASE),
    re.compile(r"معرفی\s*کن", re.IGNORECASE),
    re.compile(r"درباره\s*چیست", re.IGNORECASE),
    re.compile(r"در\s*مورد\s*چیه", re.IGNORECASE),
    re.compile(r"موضوع\s*(?:کلی|اصلی)\s*سند", re.IGNORECASE),
    re.compile(r"نکات\s*کلیدی\s*سند", re.IGNORECASE),
]

_GLOBAL_DOC_PATTERNS = [
    re.compile(r"کل\s*اسناد", re.IGNORECASE),
    re.compile(r"همه\s*اسناد", re.IGNORECASE),
    re.compile(r"تمام\s*(?:اسناد|فایل‌ها|فایلها)", re.IGNORECASE),
    re.compile(r"هر\s*دو\s*سند", re.IGNORECASE),
    re.compile(r"در\s*این\s*اسناد", re.IGNORECASE),
]

_ORDINAL_MAP = {
    "اول": 1, "یکم": 1, "نخست": 1,
    "دوم": 2,
    "سوم": 3,
    "چهارم": 4,
    "پنجم": 5,
    "ششم": 6,
    "هفتم": 7,
    "هشتم": 8,
    "نهم": 9,
    "دهم": 10
}

_FA_TO_EN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def extract_count_keyword(query: str) -> Optional[str]:
    """
    استخراج کلمه یا عبارت هدف در سوالات شمارشی (تکرار واژه).
    """
    # ۱. اگر کلمه داخل گیومه یا براکت بود
    m_quote = re.search(r'["\'«»]([^"\'«»]+)["\'«»]', query)
    if m_quote:
        val = m_quote.group(1).strip()
        if val:
            return val

    # ۲. عبارت بعد از کلمه / واژه / عبارت
    m_word = re.search(r"(?:کلمه|واژه|عبارت)\s+([^\s؟?.,!]+(?:\s+[^\s؟?.,!]+){0,2})", query)
    if m_word:
        t = m_word.group(1).strip()
        t = re.sub(r"\s+(?:تکرار|آمده|اومده|ذکر|در|چند|چقدر|شده|است|شده‌است|شده است).*$", "", t)
        if t.strip():
            return t.strip()

    # ۳. عبارت بعد از 'چند بار'
    m_times = re.search(r"چند\s*بار\s+([^\s؟?.,!]+(?:\s+[^\s؟?.,!]+){0,2})", query)
    if m_times:
        t = m_times.group(1).strip()
        t = re.sub(r"^(?:کلمه|واژه|عبارت)\s+", "", t)
        t = re.sub(r"\s+(?:تکرار|آمده|اومده|ذکر|در|استفاده|شده|است).*$", "", t)
        if t.strip():
            return t.strip()

    return None


def extract_target_doc_indices(query: str) -> List[int]:
    """
    استخراج شماره اسناد اشاره‌شده در متن سوال (مثلاً [1, 2] یا [4]).
    """
    norm_q = query.translate(_FA_TO_EN_DIGITS)
    found_indices: List[int] = []

    # استخراج ارقام مستقیم بعد از کلیدواژه‌های سند / فایل / پاسخنامه
    for m in re.finditer(r"(?:سند|فایل|پاسخنامه|شماره)\s*(?:شماره\s*)?(\d+)", norm_q):
        found_indices.append(int(m.group(1)))

    # استخراج ارقام دوگانه با واو عطف (مثلاً «۵ و ۶»)
    for m in re.finditer(r"(\d+)\s*و\s*(\d+)", norm_q):
        found_indices.extend([int(m.group(1)), int(m.group(2))])

    # استخراج کلمات ترتیبی (اول، دوم، سوم، ...)
    for word, idx in _ORDINAL_MAP.items():
        if re.search(rf"\b{word}\b", norm_q):
            found_indices.append(idx)

    return sorted(list(set(found_indices)))


def analyze_query(
    query: str,
    available_files: Optional[List[Dict[str, Any]]] = None
) -> QueryAnalysisResult:
    """
    تحلیل جامع متن سوال کاربر و تعیین استراتژی بهینه برای پاسخگویی.
    """
    norm_query = query.strip()
    is_global = any(p.search(norm_query) for p in _GLOBAL_DOC_PATTERNS)

    # ۱. اولویت اول: بررسی پرسش‌های شمارشی و تجمعی (تعداد دفعات / چند بار تکرار)
    for p in _COUNT_PATTERNS:
        if p.search(norm_query):
            kw = extract_count_keyword(norm_query)
            doc_indices = extract_target_doc_indices(norm_query)
            return QueryAnalysisResult(
                intent=QueryIntent.AGGREGATION_COUNT,
                target_keyword=kw,
                target_doc_indices=doc_indices,
                is_global=is_global or (len(doc_indices) == 0)
            )

    # ۲. اولویت دوم: بررسی پرسش‌های مقایسه‌ای
    for p in _COMPARE_PATTERNS:
        if p.search(norm_query):
            doc_indices = extract_target_doc_indices(norm_query)
            return QueryAnalysisResult(
                intent=QueryIntent.COMPARISON,
                target_doc_indices=doc_indices,
                is_global=is_global or (len(doc_indices) == 0)
            )

    # ۳. اولویت سوم: بررسی پرسش‌های خلاصه‌سازی
    for p in _SUMMARIZE_PATTERNS:
        if p.search(norm_query):
            doc_indices = extract_target_doc_indices(norm_query)
            return QueryAnalysisResult(
                intent=QueryIntent.SUMMARIZATION,
                target_doc_indices=doc_indices,
                is_global=is_global or (len(doc_indices) == 0)
            )

    # ۴. پیش‌فرض: سوالات فکتوال و نقطه‌ای عادی
    return QueryAnalysisResult(
        intent=QueryIntent.FACTUAL_POINT,
        is_global=False
    )


# ---------------------------------------------------------------------------
# موتورهای اجرای استراتژی‌های سطح سند (Document-Level Execution Engines)
# ---------------------------------------------------------------------------

def execute_keyword_count(
    chat_id: Union[int, str],
    keyword: str,
    files: Optional[List[Dict[str, Any]]] = None,
    chunks_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    محاسبه دقیق و قطعی (Deterministic) تعداد دفعات تکرار یک واژه در تمام اسناد گفتگو.
    برای جلوگیری از شمارش مضاعف حاصل از Overlap چانک‌ها، اولویت با متن تمیزشده (clean_text)
    صفحات سند از روی دیسک است.
    """
    if not keyword or not keyword.strip():
        return {"keyword": "", "total_count": 0, "by_file": [], "formatted_report": "", "citations": []}

    clean_kw = keyword.strip()
    # در زبان فارسی، واژه ممکن است با نیم‌فاصله یا بدون آن نوشته شده باشد (لپ‌تاپ vs لپ تاپ)
    kw_variants = {
        clean_kw,
        clean_kw.replace("\u200c", " "),
        clean_kw.replace(" ", "\u200c"),
        clean_kw.replace("\u200c", ""),
    }

    chat_dir = db.get_chat_data_dir(chat_id)
    chat_files = files if files is not None else db.get_files(chat_id)

    total_count = 0
    by_file: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []

    # پیمایش اسناد گفتگو
    for f in chat_files:
        fn = f.get("filename", "")
        stem = Path(fn).stem
        clean_json = chat_dir / f"{stem}_clean.json"

        file_count = 0
        pages_breakdown: Dict[int, int] = {}
        snippets: List[str] = []

        if clean_json.exists():
            try:
                with open(clean_json, "r", encoding="utf-8") as jf:
                    doc_obj = json.load(jf)
                    doc_id = doc_obj.get("doc_id", f"doc_{f.get('id', 0)}")
                    pages = doc_obj.get("pages", [])

                    for p in pages:
                        p_num = p.get("page_number", 1)
                        p_text = p.get("clean_text") or p.get("raw_text", "")

                        # شمارش رخدادها در این صفحه
                        p_matches = 0
                        for v in kw_variants:
                            p_matches = max(p_matches, p_text.count(v))

                        if p_matches > 0:
                            file_count += p_matches
                            pages_breakdown[p_num] = pages_breakdown.get(p_num, 0) + p_matches

                            # استخراج یک نمونه جمله حاوی کلمه جهت نمایش در گزارش
                            for line in p_text.splitlines():
                                if any(v in line for v in kw_variants):
                                    line_clean = line.strip()
                                    if len(line_clean) > 160:
                                        line_clean = line_clean[:160] + "..."
                                    snippets.append(f"(صفحه {p_num}): «{line_clean}»")
                                    break

                            citations.append(
                                Citation(
                                    doc_id=doc_id,
                                    filename=fn,
                                    page_number=p_num,
                                    score=1.0,
                                    snippet=f"تکرار واژه «{clean_kw}» در صفحه {p_num}"
                                ).to_dict()
                            )
            except Exception:
                pass

        # اگر از دیسک خوانده نشد، از چانک‌های وکتور دی‌بی استفاده کن
        if file_count == 0 and chunks_data and chunks_data.get("documents"):
            for doc_txt, meta in zip(chunks_data["documents"], chunks_data.get("metadatas", [])):
                if meta.get("filename") == fn or not fn:
                    c_matches = sum(doc_txt.count(v) for v in kw_variants)
                    if c_matches > 0:
                        file_count += c_matches
                        p_num = meta.get("page_number", 1)
                        pages_breakdown[p_num] = pages_breakdown.get(p_num, 0) + c_matches
                        citations.append(
                            Citation(
                                doc_id=meta.get("doc_id", "doc_unknown"),
                                filename=meta.get("filename", fn),
                                page_number=p_num,
                                score=1.0,
                                snippet=doc_txt[:120].strip() + "..."
                            ).to_dict()
                        )

        if file_count > 0:
            total_count += file_count
            by_file.append({
                "filename": fn,
                "count": file_count,
                "pages": pages_breakdown,
                "snippets": snippets[:3]
            })

    # ساخت گزارش تحلیلی به زبان فارسی
    if total_count == 0:
        report = f"واژه یا عبارت «{clean_kw}» در متن اسناد بارگذاری‌شده در این گفتگو یافت نشد."
    else:
        lines = [
            f"بر اساس بررسی دقیق کل متن اسناد، واژه یا عبارت «{clean_kw}» در مجموع **{total_count} بار** تکرار شده است:\n"
        ]
        for item in by_file:
            pg_list = ", ".join(f"صفحه {p} ({c} بار)" for p, c in sorted(item["pages"].items()))
            lines.append(f"📄 **سند `{item['filename']}`:** مجموعاً **{item['count']} بار** تکرار")
            lines.append(f"  └─ موقعیت صفحات: {pg_list}")
            if item["snippets"]:
                lines.append("  └─ نمونه جملات به کار رفته:")
                for snip in item["snippets"]:
                    lines.append(f"      • {snip}")
            lines.append("")

        report = "\n".join(lines).strip()

    return {
        "keyword": clean_kw,
        "total_count": total_count,
        "by_file": by_file,
        "formatted_report": report,
        "citations": citations
    }


def resolve_chat_files(chat_id: Union[int, str], vector_db: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    دریافت لیست فایل‌های یک گفتگو با پشتیبانی دوگانه از دیتابیس SQLite و کالکشن ChromaDB.
    """
    try:
        files = db.get_files(chat_id)
        if files:
            return files
    except Exception:
        pass

    # در صورت نبود رکورد در SQLite (مثلاً در تست‌های ایزوله)، استخراج از متادیتای موجود در ChromaDB
    if vector_db:
        try:
            chunks = vector_db.get_collection_chunks(chat_id)
            metas = chunks.get("metadatas", [])
            seen_files = {}
            for m in metas:
                fn = m.get("filename") or m.get("doc_id") or "document.pdf"
                if fn not in seen_files:
                    seen_files[fn] = {
                        "id": len(seen_files) + 1,
                        "chat_id": chat_id,
                        "filename": fn,
                        "doc_id": m.get("doc_id", "")
                    }
            return list(seen_files.values())
        except Exception:
            pass
    return []


def build_document_context_for_summary(
    chat_id: Union[int, str],
    target_files: List[Dict[str, Any]],
    max_chars: int = 16000,
    chunks_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    گردآوری کل محتوای صفحات اسناد هدف برای خلاصه‌سازی جامع، سازمان‌دهی‌شده بر اساس شماره صفحه.
    """
    chat_dir = db.get_chat_data_dir(chat_id)
    doc_blocks: List[str] = []
    citations: List[Dict[str, Any]] = []

    for f in target_files:
        fn = f.get("filename", "")
        stem = Path(fn).stem
        clean_json = chat_dir / f"{stem}_clean.json"
        file_processed = False

        if clean_json.exists():
            try:
                with open(clean_json, "r", encoding="utf-8") as jf:
                    doc_obj = json.load(jf)
                    doc_id = doc_obj.get("doc_id", f"doc_{f.get('id', 0)}")
                    pages = doc_obj.get("pages", [])

                    file_lines = [f"=== مستندات سند: {fn} (شناسه: {doc_id}) ==="]
                    for p in pages:
                        p_num = p.get("page_number", 1)
                        p_text = (p.get("clean_text") or p.get("raw_text", "")).strip()
                        if p_text:
                            file_lines.append(f"\n[سند: {fn} | شماره صفحه: {p_num}]\n{p_text}")
                            citations.append(
                                Citation(
                                    doc_id=doc_id,
                                    filename=fn,
                                    page_number=p_num,
                                    score=0.95,
                                    snippet=p_text[:120] + "..."
                                ).to_dict()
                            )

                    doc_blocks.append("\n".join(file_lines))
                    file_processed = True
            except Exception:
                pass

        # در صورت نبود فایل روی دیسک، استخراج از چانک‌های برداری موجود در حافظه
        if not file_processed and chunks_data and chunks_data.get("documents"):
            target_chunks = []
            for doc_txt, meta in zip(chunks_data["documents"], chunks_data.get("metadatas", [])):
                if meta.get("filename") == fn or meta.get("doc_id") == f.get("doc_id") or not fn:
                    target_chunks.append((meta.get("page_number", 1), meta.get("chunk_index", 0), doc_txt, meta))

            target_chunks.sort(key=lambda x: (x[0], x[1]))
            if target_chunks:
                f_lines = [f"=== مستندات سند: {fn} ==="]
                for p_num, c_idx, txt, meta in target_chunks:
                    f_lines.append(f"\n[سند: {fn} | شماره صفحه: {p_num}]\n{txt}")
                    citations.append(
                        Citation(
                            doc_id=meta.get("doc_id", "doc_unknown"),
                            filename=fn,
                            page_number=p_num,
                            score=0.95,
                            snippet=txt[:120].strip() + "..."
                        ).to_dict()
                    )
                doc_blocks.append("\n".join(f_lines))

    full_context = "\n\n---\n\n".join(doc_blocks)

    # اگر طول بیش از حد مجاز بود، کنترل بهینه انجام می‌شود
    if len(full_context) > max_chars:
        full_context = full_context[:max_chars] + "\n\n... [ادامه متن به دلیل ظرفیت خلاصه کوتاه شده است]"

    return {
        "formatted_context": full_context,
        "citations": citations[:10]  # محدود به صفحات کلیدی برای ارجاع
    }


def build_comparison_context(
    chat_id: Union[int, str],
    file_a: Dict[str, Any],
    file_b: Dict[str, Any],
    max_chars_per_doc: int = 8000,
    chunks_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    گردآوری محتوای دو سند مورد مقایسه به صورت ساختاریافته و تفکیک‌شده.
    """
    ctx_a = build_document_context_for_summary(chat_id, [file_a], max_chars=max_chars_per_doc, chunks_data=chunks_data)
    ctx_b = build_document_context_for_summary(chat_id, [file_b], max_chars=max_chars_per_doc, chunks_data=chunks_data)

    formatted_context = (
        f"### مستندات سند اول (سند مبدا الف: {file_a.get('filename', 'سند اول')}):\n"
        f"{ctx_a['formatted_context']}\n\n"
        f"============================================================\n\n"
        f"### مستندات سند دوم (سند مبدا ب: {file_b.get('filename', 'سند دوم')}):\n"
        f"{ctx_b['formatted_context']}"
    )

    all_citations = ctx_a["citations"][:4] + ctx_b["citations"][:4]
    return {
        "formatted_context": formatted_context,
        "citations": all_citations
    }

