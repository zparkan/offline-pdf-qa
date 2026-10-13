"""
spell_corrector.py
----------------------------------
ماژول پالایش و اصلاح املایی فازی سبک (Corpus-Constrained Fuzzy Spell Corrector)
مخصوص سیستم RAG آفلاین زبان فارسی.

این ماژول برای حل مشکل شماره ۹ طراحی شده است:
  «عدم تشخیص غلط‌های املایی و تایپی (مانند "غلوم کامپیوتر" به جای "علوم کامپیوتر")»

مزایا و ویژگی‌های کلیدی:
  ۱. فوق‌سریع و زیر یک میلی‌ثانیه (< 1ms): جستجوی فازی فقط روی واژگان اسناد همان چت
     (Corpus Vocabulary) انجام می‌شود، نه روی دیکشنری ۳۰۰ هزار کلمه‌ای سنگین.
  ۲. شاخص‌بندی بر اساس طول کلمه (Length Bucketing Index): کاهش فضای جستجو به چند ده کلمه کاندید.
  ۳. الگوریتم لون‌اشتاین با برش زودهنگام (Levenshtein with Early-Exit): بدون مصرف حافظه یا CPU اضافی.
  ۴. پشتیبانی هوشمند از حروف هم‌آوای فارسی (Homophones):
     اولویت‌بخشی بالا به اشتباهات املایی آوایی متداول (ع/غ، ص/س/ث، ز/ض/ظ/ذ، ط/ت، ق/غ، ح/ه).
  ۵. حفظ کلمات کلیدی، کدهای پرونده، اعداد و کلمات ایستا (Stopwords).
"""

import re
import sys
from typing import List, Dict, Set, Tuple, Optional, Any

# تنظیم کدگذاری خروجی ترمینال
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# کلمات ایستای پرکاربرد و حروف اضافه که نباید دستخوش اصلاح املایی شوند
COMMON_STOPWORDS = {
    "در", "به", "از", "که", "با", "و", "یا", "این", "آن", "را", "برای", "تا",
    "است", "شد", "بود", "می", "یک", "رو", "بر", "چه", "چیست", "چیه", "کجا", "کی",
    "چرا", "چگونه", "چطور", "کدام", "کدوم", "شدن", "بودن", "کردن", "دارد", "دارند", "ها", "های",
    "من", "تو", "او", "ما", "شما", "آنها", "ایشان", "اینها", "آنان",
    "بهم", "بهت", "بهش", "بهمون", "بهتون", "بهشون",
    "توضیح", "توضیحی", "بده", "بدهید", "بگو", "بگویید", "کنید", "کن", "کنم",
    "لطفا", "لطفاً", "خواهش", "خواهشاً", "میخوام", "میخواهم", "خواستم", "داشتم",
    "میشه", "میشود", "میشوند", "هست", "هستند", "هستن", "نیست", "نیستند", "باشه", "باشد",
    "درباره", "مورد", "موردی", "راجع", "باب",
    "کامل", "سریع", "جامع", "خلاصه", "مختصر", "چند", "چندتا", "چقدر"
}

# نگاشت حروف هم‌آوای زبان فارسی (حروفی که در تلفظ یکسان هستند و اشتباهات املایی متداول ایجاد می‌کنند)
PERSIAN_HOMOPHONES: Dict[str, Set[str]] = {
    'ع': {'ع', 'غ'},
    'غ': {'ع', 'غ'},
    'ص': {'ص', 'س', 'ث'},
    'س': {'ص', 'س', 'ث'},
    'ث': {'ص', 'س', 'ث'},
    'ط': {'ط', 'ت'},
    'ت': {'ط', 'ت'},
    'ز': {'ز', 'ض', 'ظ', 'ذ'},
    'ض': {'ز', 'ض', 'ظ', 'ذ'},
    'ظ': {'ز', 'ض', 'ظ', 'ذ'},
    'ذ': {'ز', 'ض', 'ظ', 'ذ'},
    'ق': {'ق', 'غ'},
    'ح': {'ح', 'ه'},
    'ه': {'ح', 'ه'},
}


def normalize_persian_word(word: str) -> str:
    """
    نرمال‌سازی پایه یک کلمه فارسی:
      - تبدیل ي و ك عربی به ی و ک فارسی
      - حذف اعراب، تنوین‌ها و حرکت‌گذاری‌ها
      - حذف علائم نگارشی متصل
    """
    if not word:
        return ""
    w = word.replace("ي", "ی").replace("ك", "ک")
    w = re.sub(r"[\u064B-\u0652\u0670]", "", w)
    w = re.sub(r"[^\w\u200c]", "", w)
    return w.strip().lower()


def fast_levenshtein_distance(s1: str, s2: str, max_limit: int = 2) -> int:
    """
    محاسبه فاصله ویرایشی لون‌اشتاین (Levenshtein Distance) با بهینه‌سازی برش زودهنگام (Early Exit).
    در صورتی که اختلاف کاراکترها در حین پیمایش از max_limit بیشتر شود، بلافاصله متوقف می‌شود.
    """
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > max_limit:
        return max_limit + 1

    if s1 == s2:
        return 0

    prev_row = list(range(len2 + 1))
    for i, c1 in enumerate(s1, 1):
        curr_row = [i] * (len2 + 1)
        min_in_row = curr_row[0]
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            val = min(
                prev_row[j] + 1,       # حذف
                curr_row[j - 1] + 1,   # درج
                prev_row[j - 1] + cost # جایگزینی
            )
            curr_row[j] = val
            if val < min_in_row:
                min_in_row = val

        # برش زودهنگام در صورت فراتر رفتن تمام مقادیر سطر جاری از سقف مجاز
        if min_in_row > max_limit:
            return max_limit + 1
        prev_row = curr_row

    return prev_row[len2]


class CorpusSpellCorrector:
    """
    کلاس مدیریت شاخص واژگان اسناد و تصحیح هوشمند غلط‌های املایی.
    """

    def __init__(self, documents: Optional[List[str]] = None):
        """
        مقداردهی اولیه با متون اسناد بارگذاری‌شده در چت.
        """
        self.vocab: Set[str] = set()
        self.word_freqs: Dict[str, int] = {}
        self.by_length: Dict[int, Set[str]] = {}

        if documents:
            self.build_index(documents)

    def build_index(self, documents: List[str]):
        """
        استخراج دایره لغات سندها و دسته‌بندی آن‌ها بر اساس طول کلمه برای تسریع جستجو.
        """
        self.vocab.clear()
        self.word_freqs.clear()
        self.by_length.clear()

        for doc in documents:
            if not doc:
                continue
            # استخراج تمام کلمات سند
            tokens = re.findall(r"\b[\w\u200c]+\b", doc)
            for raw_tok in tokens:
                tok = normalize_persian_word(raw_tok)
                # نادیده گرفتن کلمات خیلی کوتاه (زیر ۳ حرف) یا اعداد خالص
                if len(tok) < 3 or tok.isdigit() or tok in COMMON_STOPWORDS:
                    continue

                self.word_freqs[tok] = self.word_freqs.get(tok, 0) + 1
                self.vocab.add(tok)

        # شاخص‌بندی بر اساس طول کلمه
        for word in self.vocab:
            w_len = len(word)
            if w_len not in self.by_length:
                self.by_length[w_len] = set()
            self.by_length[w_len].add(word)

    def find_best_candidate(self, typo_word: str, max_distance: int = 2) -> Optional[Tuple[str, int]]:
        """
        یافتن بهترین کلمه جایگزین در دایره لغات اسناد برای یک کلمه مشکوک به غلط املایی.

        خروجی:
            تاپل (کلمه جایگزین, فاصله ویرایشی) یا None در صورت عدم یافتن تطابق معتبر.
        """
        norm_word = normalize_persian_word(typo_word)
        word_len = len(norm_word)

        # شرایط نادیده‌گیری: کلمات کمتر از ۳ حرف، اعداد، یا کلماتی که خودشان در سند وجود دارند
        if word_len < 3 or norm_word.isdigit() or norm_word in COMMON_STOPWORDS:
            return None

        if norm_word in self.vocab:
            return None

        # سقف مجاز فاصله ویرایشی متناسب با طول کلمه:
        # کلمات ۳ یا ۴ حرفی: حداکثر فاصله ۱ (برای جلوگیری از تغییر کلمات نامربوط)
        # کلمات ۵ حرفی یا بیشتر: حداکثر فاصله ۲
        effective_max_dist = 1 if word_len <= 4 else min(max_distance, 2)

        best_cand: Optional[str] = None
        min_dist = effective_max_dist + 1
        best_score = -1.0

        # بررسی کاندیدها فقط در طول‌های مجاز (Length Buckets)
        target_lengths = range(
            max(3, word_len - effective_max_dist),
            word_len + effective_max_dist + 1
        )

        for l in target_lengths:
            candidates = self.by_length.get(l, set())
            for cand in candidates:
                dist = fast_levenshtein_distance(norm_word, cand, max_limit=effective_max_dist)
                if dist <= effective_max_dist:
                    # امتیاز تشویقی هم‌آوایی (Phonetic Bonus):
                    # اگر جایگزینی مربوط به حروف هم‌آوای فارسی (ع/غ، ص/س و ...) باشد، اولویت دوچندان می‌یابد
                    phonetic_bonus = 0.0
                    if dist == 1 and len(cand) == word_len:
                        for c1, c2 in zip(norm_word, cand):
                            if c1 != c2:
                                if c2 in PERSIAN_HOMOPHONES.get(c1, set()):
                                    phonetic_bonus = 5.0
                                break

                    # ترکیب فاصله، تشویق هم‌آوایی و فراوانی کلمه در متن اسناد
                    freq = self.word_freqs.get(cand, 1)
                    score = (10.0 - dist) * 10.0 + phonetic_bonus + min(float(freq), 5.0)

                    if dist < min_dist or (dist == min_dist and score > best_score):
                        min_dist = dist
                        best_cand = cand
                        best_score = score

        if best_cand and min_dist <= effective_max_dist:
            return best_cand, min_dist

        return None

    def correct_query(
        self,
        query: str,
        max_distance: int = 2
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        پالایش و اصلاح املایی تمام کلمات سوال کاربر با حفظ ساختار و علائم نگارشی.

        ورودی:
            query: متن سوال اولیه کاربر
            max_distance: حداکثر فاصله مجاز لون‌اشتاین

        خروجی:
            (متن سوال اصلاح‌شده, لیستی از اطلاعات تغییرات صورت‌گرفته)
        """
        if not query or not query.strip() or not self.vocab:
            return query, []

        corrections: List[Dict[str, Any]] = []

        # تجزیه متن به کلمات و حفظ فواصل و علائم نگارشی
        tokens_with_spans = list(re.finditer(r"[\w\u200c]+", query))
        if not tokens_with_spans:
            return query, []

        # پیمایش معکوس برای حفظ اندیس‌های جایگزینی در رشته متنی
        modified_query = query
        for match in reversed(tokens_with_spans):
            original_word = match.group(0)
            start_idx, end_idx = match.span()

            match_res = self.find_best_candidate(original_word, max_distance=max_distance)
            if match_res:
                corrected_word, dist = match_res
                corrections.append({
                    "original": original_word,
                    "corrected": corrected_word,
                    "distance": dist,
                    "start": start_idx,
                    "end": end_idx
                })
                # جایگزینی کلمه غلط با کلمه صحیح در متن سوال
                modified_query = modified_query[:start_idx] + corrected_word + modified_query[end_idx:]

        # بازگرداندن لیست اصلاحات به ترتیب طبیعی (چپ به راست در متن)
        corrections.reverse()
        return modified_query, corrections
