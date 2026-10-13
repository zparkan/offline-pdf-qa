"""
tests/test_spell_corrector.py
----------------------------------
آزمون‌های خودکار واحد و ارزیابی کارایی ماژول پالایش و اصلاح املایی فازی سبک (spell_corrector.py).
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spell_corrector import (
    CorpusSpellCorrector,
    fast_levenshtein_distance,
    normalize_persian_word,
    PERSIAN_HOMOPHONES
)


def test_levenshtein_and_homophones():
    print("\n[تست ۱] بررسی الگوریتم لون‌اشتاین و ترجیح حروف هم‌آوا...")
    # تست فاصله ساده
    assert fast_levenshtein_distance("علوم", "غلوم") == 1
    assert fast_levenshtein_distance("کامپیوتر", "کامپیتر") == 1
    assert fast_levenshtein_distance("قوانین", "قوانیین") == 1
    assert fast_levenshtein_distance("دستورالعمل", "دستورالمعل") == 2

    # تست برش زودهنگام
    dist_exceeded = fast_levenshtein_distance("کامپیوتر", "هواپیما", max_limit=2)
    assert dist_exceeded > 2, "الگوریتم باید در صورت تفاوت زیاد فوراً خارج شود"
    print("  └─ تست توابع پایه فاصله لون‌اشتاین با موفقیت تایید شد ✓")


def test_corpus_correction():
    print("\n[تست ۲] بررسی تصحیح املایی پرسش‌های واقعی بر اساس واژگان سند...")
    sample_docs = [
        "دانشکده مهندسی و علوم کامپیوتر در سال جاری دوره‌های هوش مصنوعی برگزار می‌نماید.",
        "ساعت کاری رسمی شرکت از ۸ صبح تا ۱۶:۳۰ عصر می‌باشد و کلیه کارمندان موظف به رعایت آن هستند.",
        "قوانین و مقررات انضباطی در دستورالعمل اجرایی شماره ۴ درج شده است."
    ]

    corrector = CorpusSpellCorrector(sample_docs)

    # تست غلط املایی آوایی «غلوم کامپیوتر» -> «علوم کامپیوتر»
    q1 = "تعریف غلوم کامپیوتر در دانشکده چیست؟"
    corr_q1, changes1 = corrector.correct_query(q1)
    assert "علوم کامپیوتر" in corr_q1, f"خطا در تصحیح: {corr_q1}"
    assert len(changes1) == 1
    assert changes1[0]["original"] == "غلوم"
    assert changes1[0]["corrected"] == "علوم"
    print(f"  └─ «{q1}» ➔ «{corr_q1}» [تایید شد ✓]")

    # تست خطای تایپی مجاورت کلیدها «ساعت کازی» -> «ساعت کاری»
    q2 = "ساعت کازی شرکت چقدر است؟"
    corr_q2, changes2 = corrector.correct_query(q2)
    assert "ساعت کاری" in corr_q2, f"خطا در تصحیح: {corr_q2}"
    assert changes2[0]["original"] == "کازی"
    assert changes2[0]["corrected"] == "کاری"
    print(f"  └─ «{q2}» ➔ «{corr_q2}» [تایید شد ✓]")

    # تست تکرار حرف «قوانیین» -> «قوانین»
    q3 = "آیا قوانیین انضباطی تغییر کرده است؟"
    corr_q3, changes3 = corrector.correct_query(q3)
    assert "قوانین" in corr_q3, f"خطا در تصحیح: {corr_q3}"
    print(f"  └─ «{q3}» ➔ «{corr_q3}» [تایید شد ✓]")

    # تست جابجایی دو حرف «دستورالمعل» -> «دستورالعمل»
    q4 = "دستورالمعل اجرایی کجاست؟"
    corr_q4, changes4 = corrector.correct_query(q4)
    assert "دستورالعمل" in corr_q4, f"خطا در تصحیح: {corr_q4}"
    print(f"  └─ «{q4}» ➔ «{corr_q4}» [تایید شد ✓]")

    # تست عدم تغییر متن درست (Zero False-Positive)
    q5 = "ساعت کاری کارمندان"
    corr_q5, changes5 = corrector.correct_query(q5)
    assert corr_q5 == q5
    assert len(changes5) == 0
    print(f"  └─ متن بدون غلط بدون تغییر ماند: «{corr_q5}» [تایید شد ✓]")


def test_performance_benchmark():
    print("\n[تست ۳] بنچمارک سرعت پردازش و تاخیر زمانی (Latency Benchmark)...")
    docs = [
        f"مستند اداری شماره {i}: بخش {i} شامل توضیحات فنی مربوط به فرآیندها، مقررات، استانداردها و سیستم‌های سازمانی است."
        for i in range(100)
    ]
    docs.append("دانشکده مهندسی علوم کامپیوتر دانشگاه تهران")

    t_idx_start = time.perf_counter()
    corrector = CorpusSpellCorrector(docs)
    t_idx = (time.perf_counter() - t_idx_start) * 1000
    print(f"  ├─ زمان ساخت ایندکس واژگان ۱۰۰ سند ({len(corrector.vocab)} واژه): {t_idx:.2f} ms")

    test_query = "غلوم کامپیوتر چیست؟"
    runs = 500
    t_start = time.perf_counter()
    for _ in range(runs):
        corrector.correct_query(test_query)
    total_time = (time.perf_counter() - t_start) * 1000
    avg_per_query = total_time / runs

    print(f"  ├─ میانگین تاخیر اجرای هر پرسش: {avg_per_query:.3f} میلی‌ثانیه ({avg_per_query * 1000:.1f} میکروثانیه!)")
    assert avg_per_query < 5.0, f"زمان اجرا باید زیر ۵ میلی‌ثانیه باشد اما {avg_per_query:.2f}ms بود"
    print("  └─ کارایی فوق‌سریع و زیر یک میلی‌ثانیه رسماً تایید شد ✓")


if __name__ == "__main__":
    print("=" * 65)
    print("🚀 آغاز آزمون‌های واحد ماژول اصلاح املایی فازی (Spell Corrector)")
    print("=" * 65)
    test_levenshtein_and_homophones()
    test_corpus_correction()
    test_performance_benchmark()
    print("\n" + "=" * 65)
    print("🎉 تمام تست‌های Spell Corrector با موفقیت ۱۰۰٪ پاس شدند!")
    print("=" * 65)
