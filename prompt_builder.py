"""
prompt_builder.py
----------------------------------
ماژول معمار و سازنده پرامپت (Prompt Builder) برای هسته RAG آفلاین فارسی.

وظایف اصلی این ماژول:
  ۱. مهار حداکثری توهم (Anti-Hallucination Guardrails):
     تزریق دستورات محکم سیستمی برای وادار کردن مدل زبانی به پاسخ‌گویی «صرفاً بر اساس اسناد».
  ۲. سازگاری کامل با خروجی ماژول بازیاب (Retriever Compatibility):
     پذیرش مستقیم دیکشنری خروجی retriever.retrieve() یا لیستی از چانک‌ها/DocumentChunk.
  ۳. پشتیبانی از حافظه و تاریخچه گفتگو (Chat History):
     افزودن گفتگوهای پیشین چت با محدودسازی هوشمند طول تاریخچه برای صرفه‌جویی در رم و کانتکست.
  ۴. تولید خروجی چندمنظوره:
     الف) لیست استاندارد پیام‌های نقش‌دار (messages) برای llama-cpp-python
     ب) رشته خام بر اساس قالب استاندارد چت مدل‌ها (ChatML Format)
"""

import sys
from typing import List, Dict, Any, Optional, Union
from schemas import DocumentChunk

# تنظیم کدگذاری خروجی ترمینال روی UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# دستورالعمل سیستمی پیش‌فرض برای مهار توهم و هدایت دقیق مدل زبانی
DEFAULT_SYSTEM_INSTRUCTION = """تو یک دستیار هوشمند، متعهد و امانت‌دار برای پاسخ‌گویی به سوالات بر اساس اسناد شخصی کاربر هستی.

وظیفه تو این است که فقط و فقط بر اساس «زمینه‌ها و اسناد مرجع» ارائه‌شده در پیام کاربر پاسخ دهی.
حتماً قوانین زیر را بدون استثنا رعایت کن:
۱. فقط از حقایق و اطلاعات موجود در متون ارائه‌شده استفاده کن. استنتاج منطقی و تلخیص از محتوای اسناد مجاز و مطلوب است، اما هرگز از دانسته‌های بیرونی چیزی را حدس نزن یا اطلاعات ساختگی اضافه نکن.
۲. تنها در صورتی که موضوع و پاسخ سوال کلاً در اسناد ارائه‌شده پوشش داده نشده باشد، صراحتاً بگو: «پاسخی برای این پرسش در اسناد بارگذاری‌شده شما یافت نشد.»
۳. در انتهای هر ادعا یا پاسخ، حتماً نام سند و شماره صفحه منبع را ذکر کن (مثال: [سند: قرارداد.pdf، صفحه: ۳]).
۴. پاسخ باید خلاصه، مستند، منسجم، روان و کاملاً به زبان فارسی باشد."""

# پیام استاندارد در صورت عدم وجود کانتکست یا فیلتر شدن به خاطر شباهت پایین
FALLBACK_NO_CONTEXT_MESSAGE = "پاسخی برای این پرسش در اسناد بارگذاری‌شده شما یافت نشد."


class PromptBuilder:
    """
    کلاس مدیریت و ساخت پرامپت‌های مهندسی‌شده برای مدل زبانی محلی.
    """

    def __init__( 
        self,
        system_instruction: Optional[str] = None,
        max_history_turns: int = 3,
        fallback_message: str = FALLBACK_NO_CONTEXT_MESSAGE
    ):
        """
        مقداردهی اولیه سازنده پرامپت.

        ورودی:
            system_instruction: متن دستورالعمل سیستمی (در صورت None، دستور پیش‌فرض ضد توهم استفاده می‌شود).
            max_history_turns: حداکثر تعداد تبادلات قبلی کاربر-دستیار (Turn) که در پرامپت گنجانده می‌شوند.
            fallback_message: پیامی که در صورت نبود مدرک مرتبط نمایش داده می‌شود.
        """
        self.system_instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        self.max_history_turns = max(0, max_history_turns)
        self.fallback_message = fallback_message

    def format_context(
        self,
        chunks_input: Union[Dict[str, Any], List[Dict[str, Any]], List[DocumentChunk]]
    ) -> str:
        """
        تبدیل چانک‌های دریافتی به یک متن ساخت‌یافته همراه با شناسنامه منبع و شماره صفحه.

        ورودی می‌تواند:
          ۱. دیکشنری مستقیم خروجی retriever.retrieve()
          ۲. لیستی از دیکشنری‌های چانک
          ۳. لیستی از اشیای کلاس DocumentChunk
        """
        # ۱. استخراج لیست چانک‌ها بر اساس نوع ورودی
        chunks_list: List[Any] = []

        if isinstance(chunks_input, dict):
            # خروجی استاندارد ماژول بازیاب
            chunks_list = chunks_input.get("top_chunks", [])
        elif isinstance(chunks_input, list):
            chunks_list = chunks_input
        else:
            chunks_list = []

        if not chunks_list:
            return ""

        formatted_blocks = []
        for index, item in enumerate(chunks_list, start=1):
            # استخراج فیلدها با انعطاف بالا (چه شیء DocumentChunk چه دیکشنری)
            if isinstance(item, DocumentChunk):
                text = item.text
                doc_id = item.doc_id
                page_number = item.page_number
            elif isinstance(item, dict):
                text = item.get("text", "")
                # جستجو برای doc_id در سطح اصلی یا داخل metadata
                doc_id = item.get("doc_id") or item.get("metadata", {}).get("doc_id", "نامشخص")
                page_number = item.get("page_number") or item.get("metadata", {}).get("page_number", 0)
            else:
                text = str(item)
                doc_id = "نامشخص"
                page_number = 0

            # پاکسازی خطوط خالی اضافی در متن چانک
            cleaned_text = text.strip()

            header = f"[منبع شماره {index} | سند: {doc_id} | شماره صفحه: {page_number}]"
            block = f"{header}\n{cleaned_text}"
            formatted_blocks.append(block)

        # اتصال بلوک‌ها با خط جداکننده شفاف
        return "\n\n---\n\n".join(formatted_blocks)

    def format_history(
        self,
        chat_history: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        پالایش و کوتاه کردن تاریخچه گفتگو برای جلوگیری از پر شدن حافظه کانتکست مدل.

        ورودی:
            لیستی از پیام‌ها با فرمت [{"role": "user"|"assistant", "content": "..."}, ...]
        خروجی:
            لیست فیلترشده از آخرین پیام‌ها (حداکثر max_history_turns جفت پیام).
        """
        if not chat_history:
            return []

        # فیلتر پیام‌های معتبر
        valid_messages = [
            {"role": msg["role"], "content": msg["content"].strip()}
            for msg in chat_history
            if isinstance(msg, dict) and msg.get("role") in ("user", "assistant") and msg.get("content")
        ]

        # هر turn شامل یک پیام کاربر و یک پاسخ دستیار است (۲ پیام)
        max_messages = self.max_history_turns * 2
        if len(valid_messages) > max_messages:
            return valid_messages[-max_messages:]
        return valid_messages

    def build_user_message(self, query: str, formatted_context: str) -> str:
        """
        ترکیب پرسش کاربر و اسناد در قالب یک پیام ورودی منسجم و هدایت‌شده.
        """
        clean_query = query.strip()

        if not formatted_context.strip():
            # در حالتی که هیچ زمینه‌ای پیدا نشده است
            return f"پرسش کاربر:\n{clean_query}"

        return (
            "بر اساس «زمینه‌ها و اسناد مرجع» زیر، به پرسش مطرح‌شده با ذکر دقیق شماره صفحه پاسخ بده.\n\n"
            "### زمینه‌ها و اسناد مرجع:\n"
            f"{formatted_context}\n\n"
            "### پرسش کاربر:\n"
            f"{clean_query}"
        )

    def build_messages(
        self,
        query: str,
        retriever_result_or_chunks: Union[Dict[str, Any], List[Dict[str, Any]], List[DocumentChunk]],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        تولید لیست کامل پیام‌های استاندارد (System / History / User) سازگار با llama-cpp-python.

        ورودی:
            query: سوال جدید کاربر.
            retriever_result_or_chunks: خروجی ماژول بازیاب یا لیستی از چانک‌ها.
            chat_history: تاریخچه پیام‌های قبلی چت.
        """
        # ۱. پیام سیستمی ضد توهم
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_instruction}
        ]

        # ۲. افزودن تاریخچه کوتاه گفتگو (در صورت وجود)
        if chat_history:
            messages.extend(self.format_history(chat_history))

        # ۳. ساخت متن زمینه از چانک‌ها
        context_text = self.format_context(retriever_result_or_chunks)

        # ۴. افزودن پیام جدید کاربر حاوی کانتکست و پرسش
        user_content = self.build_user_message(query, context_text)
        messages.append({"role": "user", "content": user_content})

        return messages

    def build_chatml_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        تبدیل لیست پیام‌ها به رشته قالب‌بندی‌شده استاندارد ChatML (مخصوص سری Qwen و DeepSeek).
        مثال:
            <|im_start|>system
            دستور سیستمی...<|im_end|>
            <|im_start|>user
            سوال و کانتکست...<|im_end|>
            <|im_start|>assistant
        """
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # نشانه‌گذاری آغاز پاسخ مدل
        prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)

    def has_sufficient_context(
        self,
        retriever_result_or_chunks: Union[Dict[str, Any], List[Dict[str, Any]], List[DocumentChunk]]
    ) -> bool:
        """
        بررسی اینکه آیا بازیابی، محتوای معتبر و مرتبط با سوال پیدا کرده است یا خیر.
        از این تابع برای پیاده‌سازی گارد خروج سریع (Fast Exit) استفاده می‌شود.
        """
        if isinstance(retriever_result_or_chunks, dict):
            # اگر پرچم has_relevant_context در خروجی ریتریور باشد
            if "has_relevant_context" in retriever_result_or_chunks:
                return bool(retriever_result_or_chunks["has_relevant_context"])
            # در غیر این صورت وجود حداقل یک چانک را بررسی کن
            return len(retriever_result_or_chunks.get("top_chunks", [])) > 0

        elif isinstance(retriever_result_or_chunks, list):
            return len(retriever_result_or_chunks) > 0

        return False

    def rewrite_conversational_query(
        self,
        current_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        llm: Optional[Any] = None
    ) -> str:
        """
        بازنویسی هوشمند سوال برای بازیابی در مکالمات چندمرحله‌ای (Conversational Query Reformulation).
        اگر سوال کاربر ارجاعی به پیام‌های قبلی باشد (مانند: «گام‌ها رو بگو»، «بیشتر بگو»، «مورد دوم چیه»)،
        سوال را با موضوع و کلمات کلیدی پیام‌های پیشین غنی و مستقل می‌کند تا ریتریور چانک‌های دقیق را پیدا کند.
        """
        clean_query = current_query.strip()
        if not chat_history:
            return clean_query

        # آخرین پیام‌های کاربر در تاریخچه
        last_user_msgs = [m["content"] for m in chat_history if m.get("role") == "user" and m.get("content")]
        if not last_user_msgs:
            return clean_query

        last_user_query = last_user_msgs[-1].strip()

        # کلمات کلیدی نشان‌دهنده ارجاع و وابستگی به پیام قبلی
        referential_indicators = [
            "گام", "گام‌ها", "گامها", "مرحله", "مراحل", "بند", "ماده", "مورد", "موارد",
            "بیشتر", "توضیح", "کدام", "کدوم", "چرا", "این", "آن", "اینها", "آنها",
            "همان", "قبلی", "صفحه", "جزییات", "جزئیات", "چند", "چقدر", "لیست"
        ]

        words = clean_query.split()
        is_short = len(words) <= 7
        has_referential_word = any(ind in clean_query for ind in referential_indicators)

        if not (is_short or has_referential_word):
            return clean_query

        # در صورت در دسترس بودن مدل زبانی واقعی، بازنویسی سریع با LLM انجام می‌شود
        if llm is not None:
            try:
                system_prompt = (
                    "تو یک دستیار بازنویسی سوال برای جستجو در اسناد هستی. "
                    "با توجه به سوال قبلی، سوال جدید کاربر را به یک پرسش مستقل، شفاف و جستجوپذیر تبدیل کن. "
                    "فقط و فقط متن پرسش بازنویسی‌شده را بنویس و هیچ پاسخ یا توضیح اضافه‌ای نده."
                )
                user_prompt = (
                    f"سوال قبلی: {last_user_query}\n"
                    f"سوال جدید ارجاعی: {clean_query}\n"
                    f"پرسش مستقل برای جستجو:"
                )
                rewrite_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                resp = llm.create_chat_completion(
                    messages=rewrite_messages,
                    max_tokens=64,
                    temperature=0.0
                )
                choices = resp.get("choices", [])
                if choices:
                    rewritten = choices[0].get("message", {}).get("content", "").strip()
                    cleaned_rewritten = rewritten.strip(' "\'«»\n')
                    if cleaned_rewritten and len(cleaned_rewritten) > 3 and "\n" not in cleaned_rewritten:
                        return cleaned_rewritten
            except Exception:
                pass

        # روش غنی‌سازی واژگانی هوشمند (Fast Heuristic Context Enrichment)
        stopwords_query = {"چیست", "چیه", "کجاست", "کیست", "چگونه", "چطور", "چند", "آیا", "لطفا", "توضیح", "بده", "بگو", "رو", "را"}
        topic_words = [w for w in last_user_query.split() if w not in stopwords_query]
        topic_context = " ".join(topic_words[:6])

        if topic_context and topic_context not in clean_query:
            return f"{clean_query} {topic_context}"

        return clean_query
