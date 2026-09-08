# tests/test_embedder.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embedder import Embedder

def test_singleton():
    a = Embedder.get_instance()
    b = Embedder.get_instance()
    assert a is b, "Singleton شکست — دو آبجکت مختلف ساخته شد"
    print("✓ Singleton OK")

def test_embed_passages_shape():
    emb = Embedder.get_instance()
    texts = ["سلام دنیا", "این یک تست است"]
    result = emb.embed_passages(texts)
    assert result.shape == (2, 384), f"شکل اشتباه: {result.shape}"
    print(f"✓ embed_passages shape: {result.shape}")

def test_l2_normalization():
    emb = Embedder.get_instance()
    vec = emb.embed_query("تست نرمال‌سازی")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5, f"نرم L2 باید ۱ باشه، ولی {norm:.6f} شد"
    print(f"✓ L2 norm: {norm:.6f}")

def test_unload():
    emb = Embedder.get_instance()
    emb.unload_model()
    assert Embedder._instance is None, "بعد از unload باید _instance برابر None باشه"
    print("✓ unload OK")

if __name__ == "__main__":
    test_singleton()
    test_embed_passages_shape()
    test_l2_normalization()
    test_unload()
    print("\nهمه تست‌ها پاس شدن ✓")
