import os
from typing import List, Dict, Any, Union
from pathlib import Path
from sentence_transformers import SentenceTransformer

import config
from model_manager import ModelManager
from schemas import DocumentChunk


class Embedder:
    """
    ماژول تولید بردار (Embedding) برای اسناد و پرسش‌های کاربر.
    این کلاس به طور خودکار مدل را بارگذاری کرده و بردارها را برای ChromaDB آماده می‌کند.
    """

    def __init__(self, model_key: str = None):
        """
        مقداردهی اولیه و بارگذاری مدل امبدینگ در حافظه.
        
        ورودی:
            model_key (str, اختیاری): نام کلید مدل از config. اگر داده نشود،
                                     مدل فعال (config.ACTIVE_EMBEDDING) لود می‌شود.
        """
        # ۱. انتخاب مدل بر اساس ورودی یا کانفیگ سراسری
        self.model_key = model_key or config.ACTIVE_EMBEDDING
        if self.model_key not in config.EMBEDDING_MODELS:
            raise ValueError(f"مدل '{self.model_key}' در تنظیمات EMBEDDING_MODELS یافت نشد.")

        self.model_info = config.EMBEDDING_MODELS[self.model_key]
        self.dimension = self.model_info["dim"]

        # ۲. بررسی اینکه مدل روی دیسک هست یا نیاز به دانلود دارد
        manager = ModelManager(root_dir=config.MODELS_ROOT)
        success, msg, local_path = manager.get_embedding_model(
            repo_id=self.model_info["repo_id"],
            model_name=self.model_key
        )
        if not success:
            raise RuntimeError(f"خطا در آماده‌سازی فایل مدل امبدینگ: {msg}")

        # ۳. بارگذاری مدل روی حافظه رم یا کارت گرافیک (در صورت وجود)
        print(f"[*] در حال لود مدل امبدینگ '{self.model_key}' در حافظه...")
        self.model = SentenceTransformer(local_path)
        print(f"[✓] مدل با موفقیت بارگذاری شد (ابعاد بردار: {self.dimension}).")

        # ۴. تشخیص نیاز به پیشوند (برای مدل‌های خانواده e5)
        self.is_e5_family = "e5" in self.model_key.lower()

    def _format_text_for_embedding(self, text: str, is_query: bool = False) -> str:
        """
        افزودن خودکار پیشوندهای passage: یا query: در صورتی که مدل از خانواده e5 باشد.
        """
        if not self.is_e5_family:
            return text

        prefix = "query: " if is_query else "passage: "
        # اگر متن از قبل پیشوند نداشت، اضافه کن
        if not text.startswith(prefix):
            return f"{prefix}{text}"
        return text

    def embed_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32) -> Dict[str, Any]:
        """
        توضیح:
            تبدیل دسته‌ای از چانک‌های فاطمه به بردارهای عددی، آماده برای ذخیره مستقیم در ChromaDB.
        
        ورودی:
            chunks (List[DocumentChunk]): لیستی از اشیاء چانک فاطمه.
            batch_size (int): تعداد چانک‌هایی که همزمان برداری می‌شوند (پیش‌فرض ۳۲).
        
        خروجی:
            دیکشنری ۴ بخشی سازگار با متد collection.add در ChromaDB:
            {
                "ids": ["chunk_id_1", ...],
                "documents": ["متن خام چانک ۱", ...],
                "embeddings": [[0.12, ...], ...],
                "metadatas": [{"doc_id": "...", "page_number": 1, ...}, ...]
            }
        """
        if not chunks:
            return {"ids": [], "documents": [], "embeddings": [], "metadatas": []}

        # جداسازی داده‌ها و آماده‌سازی متن‌ها با پیشوند مناسب
        raw_texts = []
        ids = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            raw_texts.append(chunk.text)
            
            # ذخیره متادیتاهای کاربردی در کنار بردار
            metadatas.append({
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end
            })

        # اعمال پیشوند مدل e5 در صورت نیاز
        formatted_texts = [self._format_text_for_embedding(t, is_query=False) for t in raw_texts]

        # تولید بردارها با نرمال‌سازی کسینوسی
        vectors = self.model.encode(
            formatted_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        # تبدیل آرایه‌های نامپای به لیست پایتونی استاندارد
        embeddings_list = [v.tolist() for v in vectors]

        return {
            "ids": ids,
            "documents": raw_texts,       # متن بدون پیشوند برای نمایش تمیز در خروجی
            "embeddings": embeddings_list, # بردارهای عددی
            "metadatas": metadatas        # اطلاعات تکمیلی مثل صفحه و آیدی سند
        }

    def embed_query(self, query_text: str) -> List[float]:
        """
        توضیح:
            تبدیل سوال کاربر به یک بردار عددی تک‌بعدی برای جستجوی شباهت در ChromaDB.
        
        ورودی:
            query_text (str): متن سوال کاربر (ترجیحاً نرمال‌شده).
        
        خروجی:
            List[float]: لیست اعدادی با ابعاد مدل (مثلاً ۳۸۴ عدد اعشاری برای e5-small).
        """
        if not query_text or not query_text.strip():
            raise ValueError("متن سوال نمی‌تواند خالی باشد.")

        formatted_query = self._format_text_for_embedding(query_text, is_query=True)
        vector = self.model.encode(
            formatted_query,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return vector.tolist()
