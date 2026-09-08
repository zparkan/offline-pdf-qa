# tests/test_integration.py
"""
تست یکپارچه (Integration Test) — زینب

هدف: تست تک‌تک فایل‌ها قبلاً انجام شده (test_embedder.py و ...).
این فایل بررسی می‌کند که config.py + schemas.py + embedder.py + ChromaDB
با هم به صورت هماهنگ کار می‌کنند (یعنی جریان واقعی: چانک فاطمه -> بردار -> ذخیره -> بازیابی).

توجه: طبق درخواست، reset_db.py عمدا اینجا تست نمی‌شود.

پیش‌نیاز اجرا:
    - مدل embedding باید قبلا با save_models.py دانلود شده باشد،
      یا اتصال اینترنت برای دانلود خودکار توسط sentence-transformers موجود باشد.
    - pip install chromadb sentence-transformers numpy
"""

import os
import sys
import json
import shutil
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb

import config
from schemas import DocumentChunk
from embedder import Embedder


MOCK_CHUNKS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "mock_chunks.json"
)


def load_mock_chunks() -> list[DocumentChunk]:
    """خواندن mock_chunks.json (تحویلی فاطمه) و تبدیل به DocumentChunk."""
    with open(MOCK_CHUNKS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    chunks = [DocumentChunk.from_dict(d) for d in raw]
    assert len(chunks) > 0, "mock_chunks.json خالیه یا پیدا نشد"
    print(f"✓ {len(chunks)} چانک از mock_chunks.json خونده شد")
    return chunks


def test_config_embedder_dim_consistency():
    """
    بررسی می‌کند بردار واقعی خروجی Embedder دقیقا هم‌اندازه‌ی
    config.EMBEDDING_DIM است. اگر fail شود یعنی مدلی که config.py معرفی
    می‌کند با مدلی که واقعا لود شده یکی نیست.
    """
    emb = Embedder.get_instance()
    vec = emb.embed_query("تست ابعاد بردار")
    assert vec.shape[0] == config.EMBEDDING_DIM, (
        f"عدم تطابق ابعاد: config میگه {config.EMBEDDING_DIM}, "
        f"ولی مدل واقعی {vec.shape[0]} بعدی برگردوند"
    )
    print(f"✓ ابعاد بردار با config هماهنگه: {vec.shape[0]}")


def test_chunk_to_vector_pipeline():
    """شبیه‌سازی جریان کامل: چانک (فاطمه) -> بردار (زینب)."""
    chunks = load_mock_chunks()
    emb = Embedder.get_instance()

    texts = [c.text for c in chunks]
    vectors = emb.embed_passages(texts)

    assert vectors.shape == (len(chunks), config.EMBEDDING_DIM), (
        f"شکل نامعتبر: {vectors.shape}"
    )
    print(f"✓ {len(chunks)} چانک با موفقیت به بردار تبدیل شد: {vectors.shape}")
    return chunks, vectors


def test_chromadb_add_and_query():
    """
    بررسی نگاشت DocumentChunk -> ChromaDB.add() طبق قرارداد داده‌ای
    (data_contract.md، بخش «نگاشت DocumentChunk به ChromaDB»).
    از یک پایگاه‌داده‌ی موقت جدا استفاده می‌شود تا به chroma_db اصلی دست نخورد.
    """
    chunks = load_mock_chunks()
    emb = Embedder.get_instance()

    tmp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        client = chromadb.PersistentClient(path=tmp_dir)
        collection = client.get_or_create_collection(
            name=f"{config.CHROMA_COLLECTION_PREFIX}test_integration"
        )

        vectors = emb.embed_passages([c.text for c in chunks])

        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                }
                for c in chunks
            ],
        )

        assert collection.count() == len(chunks), "تعداد آیتم‌های اضافه‌شده با تعداد چانک‌ها یکی نیست"
        print(f"✓ {len(chunks)} چانک با موفقیت در ChromaDB موقت ذخیره شد")

        query_vec = emb.embed_query("پردازش زبان طبیعی فارسی چیست؟")
        results = collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=3,
        )

        assert len(results["ids"][0]) > 0, "هیچ نتیجه‌ای برای کوئری برنگشت"
        print(f"✓ کوئری موفق بود، {len(results['ids'][0])} چانک بازیابی شد")
        print(f"   نزدیک‌ترین چانک: {results['ids'][0][0]}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_embedder_unload_and_reload():
    """
    اطمینان از این‌که بعد از unload_model()، فراخوانی دوباره‌ی get_instance()
    یک نمونه‌ی تازه و کارا می‌سازد (سناریوی واقعی: آزادسازی رم قبل از لود LLM).
    """
    emb1 = Embedder.get_instance()
    emb1.unload_model()
    assert Embedder._instance is None, "بعد از unload باید _instance برابر None باشه"

    emb2 = Embedder.get_instance()
    vec = emb2.embed_query("تست بعد از reload")
    assert vec.shape[0] == config.EMBEDDING_DIM
    print("✓ بعد از unload و reload، امبدر دوباره درست کار می‌کنه")


if __name__ == "__main__":
    print("=== تست یکپارچه: config + schemas + embedder + ChromaDB ===\n")
    test_config_embedder_dim_consistency()
    test_chunk_to_vector_pipeline()
    test_chromadb_add_and_query()
    test_embedder_unload_and_reload()
    print("\nهمه‌ی تست‌های یکپارچه پاس شدن ✓")
