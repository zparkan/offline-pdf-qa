"""
vector_db.py
----------------------------------
ماژول مدیریت پایگاه داده برداری (ChromaDB) برای هسته اصلی سیستم RAG آفلاین.

وظایف اصلی این ماژول:
  ۱. ساخت و مدیریت ماندگار (Persistent) پایگاه داده برداری روی دیسک.
  ۲. ایجاد کالکشن مجزا و ایزوله برای هر گفتگو (با نام استاندارد chat_{chat_id}).
  ۳. ذخیره امن چانک‌های متنی ۴ قسمتی (شناسه، متن، بردار، متادیتا) با متد upsert.
  ۴. مدیریت حذف فایل‌ها (بر اساس متادیتا) و حذف کل کالکشن چت.
  ۵. ارائه کوئری برداری خام با متریک فاصله کسینوسی (Cosine) برای مصرف در ماژول بازیابی (Retriever).
  ۶. پاکسازی ایمن پایگاه داده در صورت تعویض مدل امبدینگ سراسری.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

import config

# تنظیم کدگذاری خروجی ترمینال روی UTF-8 برای جلوگیری از خطای یونیکد در ویندوز
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class VectorDB:
    """
    کلاس مدیریت پایگاه داده برداری ChromaDB.
    تمام عملیات خواندن، نوشتن، جستجو و حذف بردارها در سطح چت‌ها و فایل‌ها
    از طریق این کلاس انجام می‌شود.
    """

    def __init__(self, persist_dir: Optional[Union[str, Path]] = None):
        """
        راه‌اندازی کلاینت دیسکی (Persistent) پایگاه داده برداری.

        ورودی:
            persist_dir: مسیر پوشه ذخیره‌سازی فایل‌های پایگاه داده.
                         در صورت خالی بودن، مقدار پیش‌فرض config.DB_DIR استفاده می‌شود.
        """
        self.persist_dir = Path(persist_dir) if persist_dir else config.DB_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # ساخت کلاینت ماندگار کرومادی‌بی روی مسیر محلی (با غیرفعال‌سازی تلمتری برای سرعت و حالت کاملاً آفلاین)
        self.client: ClientAPI = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

    @staticmethod
    def _format_collection_name(chat_id: Union[int, str]) -> str:
        """
        تولید نام استاندارد برای کالکشن چت مطابق با قوانین نام‌گذاری ChromaDB:
        طول بین ۳ تا ۶۳ کاراکتر، فقط حروف، عدد، خط تیره و آندرلاین.
        """
        return f"chat_{chat_id}"

    def get_or_create_collection(self, chat_id: Union[int, str]):
        """
        دریافت کالکشن چت یا ساخت آن در صورت عدم وجود.

        نکته فنی بسیار مهم:
            متریک فاصله روی 'cosine' تنظیم شده است تا با بردار‌های نرمال‌شده
            مدل‌های SentenceTransformer سازگاری کامل و دقیق داشته باشد.
        """
        col_name = self._format_collection_name(chat_id)
        return self.client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        chat_id: Union[int, str],
        chunk_data: Dict[str, Any],
        batch_size: int = 250
    ) -> int:
        """
        ذخیره داده‌های ۴ قسمتی چانک‌ها در کالکشن چت مشخص.

        ورودی:
            chat_id: شناسه گفتگوی فعال
            chunk_data: دیکشنری خروجی متد embed_chunks شامل ۴ بخش:
                        - "ids": لیست شناسه‌های یکتا
                        - "documents": لیست متون اصلی چانک‌ها
                        - "embeddings": لیست بردارهای عددی چانک‌ها
                        - "metadatas": لیست اطلاعات جانبی (شماره صفحه، آیدی سند و...)
            batch_size: تعداد چانک‌ها در هر مرحله ارسال به دیتابیس (برای جلوگیری از خطای سرریز حافظه)

        خروجی:
            int: تعداد کل چانک‌هایی که با موفقیت ذخیره یا به‌روزرسانی شدند.
        """
        ids = chunk_data.get("ids", [])
        documents = chunk_data.get("documents", [])
        embeddings = chunk_data.get("embeddings", [])
        metadatas = chunk_data.get("metadatas", [])

        total_chunks = len(ids)
        if total_chunks == 0:
            return 0

        # دریافت یا ساخت کالکشن چت
        collection = self.get_or_create_collection(chat_id)

        # ارسال دسته‌ای (Batching) به کروما برای امنیت و پایداری
        for i in range(0, total_chunks, batch_size):
            end_idx = i + batch_size
            collection.upsert(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )

        return total_chunks

    def count_chunks(self, chat_id: Union[int, str]) -> int:
        """
        شمارش تعداد چانک‌های موجود در کالکشن یک چت.
        اگر چت کالکشنی نداشته باشد، مقدار صفر برمی‌گردد.
        """
        col_name = self._format_collection_name(chat_id)
        try:
            collection = self.client.get_collection(name=col_name)
            return collection.count()
        except Exception:
            # اگر کالکشن وجود نداشت، صفر برمی‌گرداند
            return 0

    def has_chunks(self, chat_id: Union[int, str]) -> bool:
        """
        بررسی سریع این‌که آیا در چت حداقل یک چانک وجود دارد یا خیر.
        بسیار کاربردی برای رابط کاربری تا در صورت خالی بودن به کاربر اعلام شود ابتدا فایل آپلود کند.
        """
        return self.count_chunks(chat_id) > 0

    def delete_file_chunks(self, chat_id: Union[int, str], doc_id: Union[int, str]) -> bool:
        """
        حذف تمام چانک‌های متعلق به یک سند خاص از کالکشن چت، بر اساس doc_id در متادیتا.

        ورودی:
            chat_id: شناسه چت
            doc_id: شناسه سندی که باید چانک‌هایش حذف شوند

        خروجی:
            bool: در صورت موفقیت True برمی‌گرداند.
        """
        col_name = self._format_collection_name(chat_id)
        try:
            collection = self.client.get_collection(name=col_name)
            # کروما تمام رکوردهایی که متادیتای doc_id آن‌ها برابر باشد را حذف می‌کند
            collection.delete(where={"doc_id": str(doc_id)})
            return True
        except Exception as e:
            print(f"[!] خطا یا عدم وجود کالکشن هنگام حذف فایل {doc_id} از چت {chat_id}: {e}")
            return False

    def delete_chat_collection(self, chat_id: Union[int, str]) -> bool:
        """
        حذف کامل کالکشن یک گفتگو از پایگاه داده (هنگامی که کاربر کل چت را حذف می‌کند).
        در صورتی که کالکشن اصلاً ایجاد نشده باشد، با امنیت مدیریت می‌شود و خطا رخ نمی‌دهد.
        """
        col_name = self._format_collection_name(chat_id)
        try:
            self.client.delete_collection(name=col_name)
            return True
        except Exception:
            # اگر کالکشن وجود نداشت، بدون پرتاب خطا با موفقیت فرض می‌شود
            return True

    def query_raw(
        self,
        chat_id: Union[int, str],
        query_embedding: List[float],
        top_k: int = 4,
        filter_doc_id: Optional[Union[int, str]] = None
    ) -> Dict[str, Any]:
        """
        عملیات سطح پایین جستجوی برداری (کوئری خام) برای استفاده در ماژول بازیابی (Retriever).

        ورودی:
            chat_id: شناسه چت
            query_embedding: بردار عددی سوال کاربر
            top_k: حداکثر تعداد نتایج درخواستی
            filter_doc_id: شناسه سند خاص در صورت درخواست فیلتر روی یک فایل

        خروجی:
            Dict: دیکشنری خام حاوی ids, documents, metadatas و distances
        """
        empty_result = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

        col_name = self._format_collection_name(chat_id)
        try:
            collection = self.client.get_collection(name=col_name)
        except Exception:
            return empty_result

        total_count = collection.count()
        if total_count == 0:
            return empty_result

        # تعداد نتایج نباید بیشتر از کل چانک‌های موجود باشد
        n_results = min(top_k, total_count)

        # ساخت فیلتر در صورت مشخص شدن فایل خاص
        where_clause = {"doc_id": str(filter_doc_id)} if filter_doc_id else None

        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause
        )

    def clear_all_collections(self) -> int:
        """
        پاکسازی تمام کالکشن‌های موجود در پایگاه داده برداری.
        این تابع به عنوان مرجع اصلی پاکسازی توسط config.clear_all_databases فراخوانی می‌شود
        تا از قفل شدن فایل‌های دیتابیس در ویندوز جلوگیری شود.

        خروجی:
            int: تعداد کالکشن‌های حذف شده
        """
        collections = self.client.list_collections()
        deleted_count = 0
        for col in collections:
            try:
                # پشتیبانی از تفاوت ورژن‌های شیء Collection یا رشته در Chroma
                col_name = col.name if hasattr(col, "name") else str(col)
                self.client.delete_collection(name=col_name)
                deleted_count += 1
            except Exception as e:
                print(f"[!] خطا در حذف کالکشن: {e}")
        return deleted_count


# =====================================================================
# بخش تست مستقل ماژول (Self-Test)
# =====================================================================
if __name__ == "__main__":
    print("[*] تست صحت عملکرد ماژول VectorDB...")

    vdb = VectorDB()

    test_chat_id = "test_999"
    # ۱. ایجاد کالکشن خالی
    col = vdb.get_or_create_collection(test_chat_id)
    print(f"[✓] کالکشن ساخته شد. تعداد اولیه چانک‌ها: {vdb.count_chunks(test_chat_id)}")
    assert vdb.has_chunks(test_chat_id) is False

    # ۲. داده آزمایشی ۴ قسمتی (شبیه‌ساز خروجی embedder)
    mock_data = {
        "ids": ["docA_c001", "docA_c002", "docB_c001"],
        "documents": [
            "قوانین حضور و غیاب در شرکت",
            "ساعات کاری منعطف از ساعت ۸ تا ۱۰ صبح",
            "دستورالعمل بیمه تکمیلی پرسنل"
        ],
        "embeddings": [
            [0.1] * 384,
            [0.2] * 384,
            [0.9] * 384
        ],
        "metadatas": [
            {"doc_id": "docA", "page_number": 1},
            {"doc_id": "docA", "page_number": 2},
            {"doc_id": "docB", "page_number": 1}
        ]
    }

    # ۳. افزودن چانک‌ها
    added = vdb.add_chunks(test_chat_id, mock_data)
    print(f"[✓] تعداد چانک‌های ذخیره‌شده: {added}")
    assert vdb.count_chunks(test_chat_id) == 3
    assert vdb.has_chunks(test_chat_id) is True

    # ۴. تست جستجوی برداری خام
    query_vec = [0.15] * 384
    results = vdb.query_raw(test_chat_id, query_vec, top_k=2)
    print(f"[✓] نتایج جستجوی خام یافت شد: {len(results['ids'][0])} مورد.")

    # ۵. تست جستجو با فیلتر فایل خاص
    filtered_results = vdb.query_raw(test_chat_id, query_vec, top_k=2, filter_doc_id="docB")
    print(f"[✓] جستجو با فیلتر docB: {filtered_results['ids'][0]}")
    assert filtered_results['ids'][0] == ["docB_c001"]

    # ۶. تست حذف چانک‌های یک فایل
    vdb.delete_file_chunks(test_chat_id, "docA")
    print(f"[✓] تعداد چانک‌ها پس از حذف docA: {vdb.count_chunks(test_chat_id)}")
    assert vdb.count_chunks(test_chat_id) == 1

    # ۷. تست حذف کل کالکشن چت
    vdb.delete_chat_collection(test_chat_id)
    print(f"[✓] کالکشن چت با موفقیت حذف شد. تعداد چانک‌ها: {vdb.count_chunks(test_chat_id)}")
    assert vdb.has_chunks(test_chat_id) is False

    print("\n[🎉] تمام تست‌های پایگاه داده برداری با موفقیت ۱۰۰٪ پاس شدند!")
