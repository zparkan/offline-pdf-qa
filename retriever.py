"""
retriever.py
----------------------------------
ماژول بازیابی ترکیبی (Hybrid Retriever) برای هسته اصلی سیستم RAG آفلاین.

ویژگی‌های کلیدی این ماژول:
  ۱. بازیابی معنایی (Dense Retrieval): با استفاده از Embedder و جستجوی برداری کسینوسی در ChromaDB.
  ۲. بازیابی واژگانی (Sparse / Lexical Retrieval): با الگوریتم استاندارد BM25 (Okapi) روی متن چانک‌های چت.
  ۳. ادغام رتبه‌ها بدون مدل (Zero-Model Fusion): با الگوریتم RRF (Reciprocal Rank Fusion).
  ۴. پشتیبانی کامل از فیلتر اسناد: امکان محدودسازی جستجو به یک یا چند doc_id خاص در گفتگوی فعلی.
  ۵. خروجی غنی و آماده برای ماژول Prompt Builder: شامل متن چانک، شماره صفحه، شناسه سند، امتیاز و بردار سوال.
"""

import re
import sys
import time
from typing import List, Dict, Any, Optional, Union
from rank_bm25 import BM25Okapi

import config
from embedder import Embedder
from vector_db import VectorDB

# تنظیم کدگذاری خروجی ترمینال روی UTF-8 برای جلوگیری از خطای یونیکد در ویندوز
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass



# لیست حروف ربط و کلمات ایستای پرکاربرد فارسی (Stopwords) برای جلوگیری از تطابق اشتباه در BM25
PERSIAN_STOPWORDS = {
    "در", "به", "از", "که", "با", "و", "یا", "این", "آن", "را", "برای", "تا",
    "است", "شد", "بود", "می", "یک", "رو", "بر", "چه", "چیست", "کجا", "کی",
    "چرا", "چگونه", "کدام", "شدن", "بودن", "کردن", "دارد", "دارند", "ها", "های"
}


class Retriever:
    """
    موتور بازیابی ترکیبی (Hybrid Retriever) مبتنی بر Dense + BM25 + RRF.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_db: Optional[VectorDB] = None,
        rrf_k: int = 60
    ):
        """
        مقداردهی اولیه ماژول بازیاب.

        ورودی:
            embedder: نمونه کلاس Embedder (در صورت None، یک نمونه خودکار ساخته می‌شود).
            vector_db: نمونه کلاس VectorDB (در صورت None، نمونه پیش‌فرض ایجاد می‌شود).
            rrf_k: ضریب ثابت الگوریتم RRF (پیش‌فرض استاندارد علمی: ۶۰).
        """
        self.embedder = embedder or Embedder()
        self.vector_db = vector_db or VectorDB()
        self.rrf_k = rrf_k

    @staticmethod
    def tokenize_persian(text: str, remove_stopwords: bool = True) -> List[str]:
        """
        توکنایز و نرمال‌سازی سریع کلمات فارسی برای الگوریتم BM25.
        
        این تابع:
          - حروف عربی 'ي' و 'ك' را به فارسی تبدیل می‌کند.
          - اعراب، تنوین‌ها و نشانه‌های نگارشی اضافه را حذف می‌کند.
          - کلمات ایستای پرکاربرد (حروف ربط و اضافه مثل «در»، «به»، «از») را فیلتر می‌کند.
          - کلمات معنادار را به صورت توکن‌های مجزا برمی‌گرداند.
        """
        if not text:
            return []

        # ۱. یکپارچه‌سازی حروف عربی و فارسی
        normalized = text.replace("ي", "ی").replace("ك", "ک")

        # ۲. حذف تنوین‌ها و حرکت‌گذاری‌ها
        normalized = re.sub(r"[\u064B-\u0652]", "", normalized)

        # ۳. استخراج کلمات و اعداد فارسی و انگلیسی (نادیده گرفتن علائم نگارشی مثل !، . ، ؟)
        tokens = re.findall(r"[\w\u0600-\u06FF]+", normalized.lower())

        if remove_stopwords:
            filtered = [t for t in tokens if t not in PERSIAN_STOPWORDS]
            return filtered if filtered else tokens

        return tokens

    def retrieve(
        self,
        query_text: str,
        chat_id: Union[int, str],
        top_k: int = 4,
        filter_doc_ids: Optional[Union[int, str, List[Union[int, str]]]] = None,
        min_similarity: float = 0.70,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        عملیات اصلی بازیابی ترکیبی برای یک سوال در گفتگوی مشخص.

        ورودی:
            query_text: متن سوال کاربر
            chat_id: شناسه چت فعال
            top_k: تعداد نهایی مرتبط‌ترین چانک‌های خروجی
            filter_doc_ids: یک شناسه فایل یا لیستی از شناسه‌های فایل برای فیلتر
            min_similarity: حداقل آستانه شباهت معنایی برای تشخیص مرتبط بودن پاسخ (پیش‌فرض: 0.30)
            verbose: نمایش لاگ‌های وضعیت و زمان‌سنجی هر مرحله

        خروجی:
            Dict حاوی:
              - "query_text": متن سوال
              - "query_embedding": بردار سوال (آماده برای جلوگیری از محاسبه مجدد)
              - "top_chunks": لیستی از چانک‌های برگزیده مرتب‌شده با اطلاعات کامل
              - "has_relevant_context": بولین (آیا اطلاعات مرتبطی در اسناد کاربر وجود دارد یا خیر)
              - "max_similarity": بیشترین شباهت معنایی به دست آمده
              - "stats": آمار عملکرد و زمان اجرای میلی‌ثانیه‌ای
        """
        start_time = time.perf_counter()

        # بررسی اعتبار ورودی سوال
        if not query_text or not query_text.strip():
            raise ValueError("متن سوال کاربر نمی‌تواند خالی باشد.")

        query_text = query_text.strip()
        if verbose:
            print(f"\n[Retriever] 🔍 شروع بازیابی ترکیبی برای سوال: «{query_text}» (چت: {chat_id})")

        # ۱. بررسی وجود چانک در چت فعلی
        chat_total_chunks = self.vector_db.count_chunks(chat_id)
        if chat_total_chunks == 0:
            if verbose:
                print(f"[Retriever] ⚠️ هیچ چانکی در چت '{chat_id}' یافت نشد.")
            return {
                "query_text": query_text,
                "query_embedding": [],
                "top_chunks": [],
                "has_relevant_context": False,
                "max_similarity": 0.0,
                "stats": {
                    "chat_id": str(chat_id),
                    "total_chunks_available": 0,
                    "elapsed_ms": round((time.perf_counter() - start_time) * 1000, 2)
                }
            }

        # تعیین عمق استخراج کاندیدهای اولیه (fetch_k) برای اثرگذاری مناسب RRF
        fetch_k = min(max(top_k * 3, 10), chat_total_chunks)

        # ---------------------------------------------------------
        # فاز ۱: بازیابی معنایی (Dense Retrieval)
        # ---------------------------------------------------------
        t_dense_start = time.perf_counter()
        if verbose:
            print(f"[Retriever] 🧠 فاز ۱: تولید بردار سوال و جستجوی برداری کسینوسی...")

        query_embedding = self.embedder.embed_query(query_text)

        dense_raw = self.vector_db.query_raw(
            chat_id=chat_id,
            query_embedding=query_embedding,
            top_k=fetch_k,
            filter_doc_ids=filter_doc_ids
        )

        dense_ids = dense_raw.get("ids", [[]])[0]
        dense_docs = dense_raw.get("documents", [[]])[0]
        dense_metas = dense_raw.get("metadatas", [[]])[0]
        dense_distances = dense_raw.get("distances", [[]])[0] if "distances" in dense_raw else []

        # نگاشت اطلاعات چانک‌ها بر اساس شناسه و رتبه معنایی
        chunk_store: Dict[str, Dict[str, Any]] = {}
        dense_ranks: Dict[str, int] = {}

        for rank, cid in enumerate(dense_ids, start=1):
            idx = rank - 1
            meta = dense_metas[idx] if idx < len(dense_metas) else {}
            doc_text = dense_docs[idx] if idx < len(dense_docs) else ""
            dist = dense_distances[idx] if idx < len(dense_distances) else None

            # محاسبه درصد شباهت کسینوسی (Similarity = 1 - Cosine Distance)
            sim = round(max(0.0, 1.0 - dist), 4) if dist is not None else None

            dense_ranks[cid] = rank
            chunk_store[cid] = {
                "chunk_id": cid,
                "text": doc_text,
                "doc_id": meta.get("doc_id", ""),
                "page_number": meta.get("page_number", 0),
                "distance": dist,
                "similarity": sim,
                "metadata": meta,
                "dense_rank": rank,
                "bm25_rank": None
            }

        t_dense = (time.perf_counter() - t_dense_start) * 1000
        if verbose:
            print(f"[Retriever]  └─ {len(dense_ranks)} کاندید معنایی در {t_dense:.1f} میلی‌ثانیه یافت شد.")

        # ---------------------------------------------------------
        # فاز ۲: بازیابی واژگانی (Sparse Retrieval با BM25)
        # ---------------------------------------------------------
        t_bm25_start = time.perf_counter()
        if verbose:
            print(f"[Retriever] 📖 فاز ۲: ساخت ایندکس BM25 و جستجوی واژگانی کلمات کلیدی...")

        # دریافت تمام چانک‌های مجاز چت (با اعمال فیلتر اسناد)
        all_chunks_data = self.vector_db.get_collection_chunks(
            chat_id=chat_id,
            filter_doc_ids=filter_doc_ids
        )

        corpus_ids = all_chunks_data.get("ids", [])
        corpus_docs = all_chunks_data.get("documents", [])
        corpus_metas = all_chunks_data.get("metadatas", [])

        bm25_ranks: Dict[str, int] = {}

        if corpus_docs:
            tokenized_corpus = [self.tokenize_persian(doc) for doc in corpus_docs]
            query_tokens = self.tokenize_persian(query_text)

            if query_tokens:
                bm25 = BM25Okapi(tokenized_corpus)
                bm25_scores = bm25.get_scores(query_tokens)

                # مرتب‌سازی نتایج بر اساس نمره BM25 نزولی
                scored_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True
                )

                # انتخاب حداکثر fetch_k نتیجه برتر که نمره بزرگتر از صفر دارند
                valid_rank = 1
                for idx in scored_indices:
                    # فقط چانک‌هایی که حداقل یک تطابق واژگانی داشته باشند
                    if bm25_scores[idx] <= 0:
                        break

                    cid = corpus_ids[idx]
                    bm25_ranks[cid] = valid_rank

                    # اگر چانک قبلاً در جستجوی معنایی نیامده بود، به انبار اضافه کن
                    if cid not in chunk_store:
                        meta = corpus_metas[idx] if idx < len(corpus_metas) else {}
                        chunk_store[cid] = {
                            "chunk_id": cid,
                            "text": corpus_docs[idx],
                            "doc_id": meta.get("doc_id", ""),
                            "page_number": meta.get("page_number", 0),
                            "distance": None,
                            "similarity": None,
                            "metadata": meta,
                            "dense_rank": None,
                            "bm25_rank": valid_rank
                        }
                    else:
                        chunk_store[cid]["bm25_rank"] = valid_rank

                    valid_rank += 1
                    if valid_rank > fetch_k:
                        break

        t_bm25 = (time.perf_counter() - t_bm25_start) * 1000
        if verbose:
            print(f"[Retriever]  └─ {len(bm25_ranks)} کاندید واژگانی در {t_bm25:.1f} میلی‌ثانیه یافت شد.")

        # ---------------------------------------------------------
        # فاز ۳: ادغام رتبه‌ها با فرمول RRF (Reciprocal Rank Fusion)
        # ---------------------------------------------------------
        t_rrf_start = time.perf_counter()
        if verbose:
            print(f"[Retriever] ⚖️ فاز ۳: ادغام رتبه‌ها با فرمول RRF (k={self.rrf_k})...")

        scored_chunks: List[Dict[str, Any]] = []

        for cid, item in chunk_store.items():
            r_dense = dense_ranks.get(cid)
            r_bm25 = bm25_ranks.get(cid)

            # فرمول RRF: 1 / (k + rank)
            rrf_score = 0.0
            if r_dense is not None:
                rrf_score += 1.0 / (self.rrf_k + r_dense)
            if r_bm25 is not None:
                rrf_score += 1.0 / (self.rrf_k + r_bm25)

            item["score"] = round(rrf_score, 6)
            scored_chunks.append(item)

        # مرتب‌سازی نهایی بر اساس بیشترین نمره RRF
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)

        # برش بر اساس top_k
        top_chunks = scored_chunks[:top_k]

        # ---------------------------------------------------------
        # فاز ۴: اعتبارسنجی ارتباط پاسخ (Relevance Thresholding)
        # ---------------------------------------------------------
        # محاسبه بیشترین شباهت معنایی موجود در نتایج برتر
        max_similarity = 0.0
        for c in top_chunks:
            if c.get("similarity") is not None and c["similarity"] > max_similarity:
                max_similarity = c["similarity"]

        # بررسی وجود تطابق واژگانی (BM25) در چانک‌های برتر
        has_bm25_match = any(c.get("bm25_rank") is not None for c in top_chunks)

        # اگر شباهت کمتر از آستانه بود و هیچ کلمه کلیدی هم مچ نشد -> پاسخ در اسناد نیست
        has_relevant_context = (max_similarity >= min_similarity) or has_bm25_match

        total_elapsed = (time.perf_counter() - start_time) * 1000
        if verbose:
            status_text = "مرتبط است ✓" if has_relevant_context else "نامرتبط با اسناد ❌"
            print(f"[Retriever] ✅ بازیابی موفق: {len(top_chunks)} چانک برتر در {total_elapsed:.1f} میلی‌ثانیه انتخاب شدند. وضعیت ارتباط: {status_text} (بیشترین شباهت: {max_similarity})")
            for i, chunk in enumerate(top_chunks, start=1):
                d_rank = f"D#{chunk['dense_rank']}" if chunk['dense_rank'] else "D#--"
                b_rank = f"B#{chunk['bm25_rank']}" if chunk['bm25_rank'] else "B#--"
                sim_str = f"شباهت: {chunk['similarity']}" if chunk['similarity'] is not None else "شباهت: --"
                print(f"      {i}. [{chunk['chunk_id']}] (صفحه {chunk['page_number']}) | امتیاز RRF: {chunk['score']} ({d_rank}, {b_rank}) | {sim_str}")

        return {
            "query_text": query_text,
            "query_embedding": query_embedding,
            "top_chunks": top_chunks,
            "has_relevant_context": has_relevant_context,
            "max_similarity": max_similarity,
            "stats": {
                "chat_id": str(chat_id),
                "total_chunks_available": chat_total_chunks,
                "filtered_doc_ids": filter_doc_ids,
                "dense_candidates_count": len(dense_ranks),
                "bm25_candidates_count": len(bm25_ranks),
                "dense_time_ms": round(t_dense, 2),
                "bm25_time_ms": round(t_bm25, 2),
                "total_elapsed_ms": round(total_elapsed, 2)
            }
        }



