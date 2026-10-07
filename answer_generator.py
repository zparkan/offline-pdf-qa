"""
answer_generator.py
----------------------------------
ماژول تولید پاسخ (Answer Generator) برای هسته اصلی سیستم RAG آفلاین فارسی.

ویژگی‌های کلیدی این ماژول:
  ۱. تولید پاسخ استنادی به صورت استریم زنده (Streaming Tokens):
     ارسال کلمه به کلمه پاسخ به رابط کاربری (UI) برای تجربه کاربری سریع و مشابه ChatGPT.
  ۲. ارسال مراجع و استنادها (Citations):
     استخراج شناسنامه دقیق منبع (نام سند، شماره صفحه، نمره شباهت و برش متن) در انتهای پاسخ.
  ۳. گارد خروج سریع (Fast Exit):
     در صورت نامرتبط بودن سوال با اسناد، تولید فوری پیام بدون درگیر کردن مدل زبانی (زیر ۱۰۰ میلی‌ثانیه).
  ۴. سازگاری کامل با قرارداد داده‌ای فرانت‌اند (ui_backend_contract.md):
     پیاده‌سازی تابع get_rag_response_stream به صورت AsyncGenerator.
  ۵. پشتیبانی از موتور Llama (فایل‌های GGUF) به همراه حالت هوشمند شبیه‌ساز (Mock Mode):
     برای تضمین اجرای روان تست‌ها و توسعه حتی در صورت نبود فایل حجیم مدل محلی.
"""

import os
import sys
import asyncio
from typing import List, Dict, Any, Optional, Union, Generator, AsyncGenerator
from pathlib import Path

# تنظیم کدگذاری خروجی ترمینال روی UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from retriever import Retriever
from prompt_builder import PromptBuilder

# بارگذاری ساختارهای داده‌ای فرانت‌اند
try:
    from gharardad_dade.ui_schemas import Citation, StreamChunk
except ImportError:
    # ساختار جایگزین در صورت عدم دسترسی به پوشه قرارداد
    from dataclasses import dataclass, asdict

    @dataclass
    class Citation:
        doc_id: str
        filename: str
        page_number: int
        score: float
        snippet: str

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)

    @dataclass
    class StreamChunk:
        type: str
        content: Optional[str] = None
        citations: Optional[List[Any]] = None

        def to_dict(self) -> Dict[str, Any]:
            res: Dict[str, Any] = {"type": self.type}
            if self.content is not None:
                res["content"] = self.content
            if self.citations is not None:
                res["citations"] = [c.to_dict() if hasattr(c, "to_dict") else c for c in self.citations]
            return res

# تلاش برای وارد کردن کتابخانه llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False


class AnswerGenerator:
    """
    موتور استنتاج و تولید پاسخ نهایی RAG بر پایه مدل زبانی محلی GGUF و ریتریور ترکیبی.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        n_ctx: int = 2048,
        mock_mode: bool = False,
        auto_download: bool = False
    ):
        """
        مقداردهی اولیه ماژول تولید پاسخ.

        ورودی:
            retriever: نمونه کلاس بازیاب (در صورت None به صورت خودکار ساخته می‌شود).
            prompt_builder: نمونه کلاس سازنده پرامپت (در صورت None نمونه پیش‌فرض ساخته می‌شود).
            model_name: کلید مدل زبانی از config (پیش‌فرض: config.ACTIVE_LLM).
            temperature: دمای تولید متن (پیش‌فرض ۰.۱ برای حداقل توهم و دقت استنادی بالا).
            max_tokens: سقف تعداد توکن‌های تولیدی.
            n_ctx: اندازه پنجره زمینه مدل زبانی.
            mock_mode: فعال‌سازی دستی حالت شبیه‌ساز (برای تست‌های سریع بدون بارگذاری مدل).
        """
        self.retriever = retriever or Retriever()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.model_name = model_name or getattr(config, "ACTIVE_LLM", "qwen-1.5b")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self.mock_mode = mock_mode
        self.auto_download = auto_download

        self.llm = None
        self.is_loaded = False

        # بارگذاری اولیه مدل در صورت فعال نبودن mock_mode دستی
        if not self.mock_mode:
            self._load_llm()

    def _load_llm(self) -> bool:
        """
        بارگذاری مدل زبانی کوانتیزه‌شده از پوشه models/llm با استفاده از llama-cpp-python.
        در صورت عدم وجود کتابخانه یا فایل مدل، سیستم به طور خودکار به حالت Mock سوئیچ می‌کند.
        """
        if not LLAMA_AVAILABLE:
            print("[AnswerGenerator] ⚠️ کتابخانه llama-cpp-python نصب نیست. حالت شبیه‌ساز (Mock Mode) فعال شد.")
            self.mock_mode = True
            return False

        model_info = getattr(config, "LLM_MODELS", {}).get(self.model_name)
        if not model_info:
            print(f"[AnswerGenerator] ⚠️ مدل '{self.model_name}' در config.LLM_MODELS یافت نشد. فعال‌سازی Mock Mode.")
            self.mock_mode = True
            return False

        filename = model_info.get("filename")
        repo_id = model_info.get("repo")
        llm_dir = getattr(config, "LLM_DIR", Path("models/llm"))
        model_path = llm_dir / filename

        if not model_path.exists():
            if self.auto_download:
                print(f"[AnswerGenerator] ⚠️ فایل وزن مدل در '{model_path}' یافت نشد. در حال دریافت خودکار...")
                from model_manager import ModelManager
                manager = ModelManager(root_dir=getattr(config, "MODELS_ROOT", "models"))
                success, msg, local_path = manager.setup_llm(
                    mode="local",
                    model_name_or_repo=repo_id,
                    filename=filename
                )
                if success and local_path and Path(local_path).exists():
                    model_path = Path(local_path)
                else:
                    print(f"[AnswerGenerator] ⚠️ امکان بارگذاری یا دانلود فایل مدل فراهم نشد ({msg}). فعال‌سازی Mock Mode.")
                    self.mock_mode = True
                    return False
            else:
                print(f"[AnswerGenerator] ℹ️ فایل وزن مدل در '{model_path}' موجود نیست. فعال‌سازی Mock Mode.")
                self.mock_mode = True
                return False

        try:
            # محاسبه تعداد بهینه ترد‌های پردازنده
            threads = max(1, (os.cpu_count() or 4) - 1)
            print(f"[AnswerGenerator] ⏳ در حال بارگذاری مدل '{self.model_name}' از '{model_path}'...")
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=self.n_ctx,
                n_threads=threads,
                verbose=False
            )
            self.is_loaded = True
            print(f"[AnswerGenerator] ✅ مدل زبانی '{self.model_name}' با موفقیت لود شد.")
            return True
        except Exception as e:
            print(f"[AnswerGenerator] ❌ خطا در بارگذاری مدل: {e}. سوییچ به حالت شبیه‌ساز.")
            self.mock_mode = True
            return False

    def _extract_citations(self, top_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        استخراج مراجع و استنادها از چانک‌های برتر برای ارسال به رابط کاربری.
        """
        citations: List[Dict[str, Any]] = []
        for chunk in top_chunks:
            doc_id = chunk.get("doc_id") or chunk.get("metadata", {}).get("doc_id", "نامشخص")
            filename = chunk.get("filename") or chunk.get("metadata", {}).get("filename") or str(doc_id)
            page_number = chunk.get("page_number") or chunk.get("metadata", {}).get("page_number", 0)
            score = chunk.get("similarity") if chunk.get("similarity") is not None else chunk.get("score", 0.0)
            text = chunk.get("text", "")
            snippet = (text[:140] + "...") if len(text) > 140 else text

            citation_obj = Citation(
                doc_id=str(doc_id),
                filename=str(filename),
                page_number=int(page_number),
                score=round(float(score), 3),
                snippet=snippet.strip()
            )
            citations.append(citation_obj.to_dict())

        return citations

    def generate_stream(
        self,
        query: str,
        chat_id: Union[int, str],
        filter_doc_ids: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        تولید استریم همگام (Synchronous Generator) پاسخ و مراجع.
        """
        # ۱. بازیابی چانک‌های مرتبط با ریتریور ترکیبی
        retrieval_res = self.retriever.retrieve(
            query_text=query,
            chat_id=chat_id,
            filter_doc_ids=filter_doc_ids
        )

        top_chunks = retrieval_res.get("top_chunks", [])

        # ۲. گارد خروج سریع (Fast Exit): آیا مدرک معتبری وجود دارد؟
        if not self.prompt_builder.has_sufficient_context(retrieval_res):
            fallback_msg = self.prompt_builder.fallback_message
            for word in fallback_msg.split(" "):
                yield StreamChunk(type="token", content=word + " ").to_dict()

            # ارسال مراجع خالی
            yield StreamChunk(type="sources", citations=[]).to_dict()
            return

        # ۳. آماده‌سازی استنادها از چانک‌های برتر
        citations = self._extract_citations(top_chunks)

        # ۴. ساخت پیام‌های پرامپت استاندارد
        messages = self.prompt_builder.build_messages(
            query=query,
            retriever_result_or_chunks=retrieval_res,
            chat_history=chat_history
        )

        # ۵. تولید توکن‌ها (مدل واقعی یا شبیه‌ساز)
        if not self.mock_mode and self.llm is not None:
            try:
                response_stream = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=True
                )
                for chunk in response_stream:
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield StreamChunk(type="token", content=token).to_dict()
            except Exception as e:
                err_msg = f"⚠️ خطا در تولید پاسخ با مدل زبانی: {e}"
                yield StreamChunk(type="token", content=err_msg).to_dict()
        else:
            # حالت شبیه‌ساز (Mock Mode): در صورتی که فایل وزن مدل هنوز دانلود نشده باشد
            best_chunk_text = top_chunks[0].get("text", "") if top_chunks else ""
            mock_reply = (
                f"⚠️ [توجه: مدل زبانی هنوز دانلود نشده و سیستم در حالت شبیه‌ساز است]\n\n"
                f"بر اساس اسناد ارائه‌شده (صفحه {top_chunks[0].get('page_number', 1)}):\n"
                f"{best_chunk_text.strip()}"
            )
            for word in mock_reply.split(" "):
                yield StreamChunk(type="token", content=word + " ").to_dict()

        # ۶. ارسال نهایی مراجع و استنادها
        yield StreamChunk(type="sources", citations=citations).to_dict()

    async def get_rag_response_stream(
        self,
        query: str,
        chat_id: Union[int, str],
        filter_doc_ids: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        تولید استریم ناهمگام (AsyncGenerator) هماهنگ با قرارداد رسمی فرانت‌اند با بک‌اند.
        """
        # اجرای تولید به صورت استریم زنده
        loop = asyncio.get_event_loop()
        gen = self.generate_stream(
            query=query,
            chat_id=chat_id,
            filter_doc_ids=filter_doc_ids,
            chat_history=chat_history
        )

        # تبدیل ژنراتور همگام به استریم ناهمگام برای فرانت‌اند
        for item in gen:
            yield item
            await asyncio.sleep(0.01)

    def generate_response(
        self,
        query: str,
        chat_id: Union[int, str],
        filter_doc_ids: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        تولید پاسخ یکپارچه (غیر استریمی) برای تست‌ها و استفاده در سایر ماژول‌ها.
        """
        answer_parts: List[str] = []
        citations: List[Dict[str, Any]] = []

        for chunk in self.generate_stream(
            query=query,
            chat_id=chat_id,
            filter_doc_ids=filter_doc_ids,
            chat_history=chat_history
        ):
            if chunk.get("type") == "token":
                answer_parts.append(chunk.get("content", ""))
            elif chunk.get("type") == "sources":
                citations = chunk.get("citations", [])

        full_answer = "".join(answer_parts).strip()
        has_context = len(citations) > 0

        return {
            "query": query,
            "answer": full_answer,
            "citations": citations,
            "has_relevant_context": has_context,
            "model_name": self.model_name
        }


# =====================================================================
# توابع مستقل سطح ماژول جهت اتصال مستقیم به رابط کاربری مشکات
# =====================================================================

_default_generator_instance: Optional[AnswerGenerator] = None

def get_answer_generator() -> AnswerGenerator:
    """دریافت نمونه اشتراکی AnswerGenerator (الگوی Singleton)"""
    global _default_generator_instance
    if _default_generator_instance is None:
        _default_generator_instance = AnswerGenerator()
    return _default_generator_instance


async def get_rag_response_stream(
    query: str,
    chat_id: int,
    filter_doc_ids: Optional[List[str]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    تابع ارکستراتور اصلی تعریف‌شده در ui_backend_contract.md.
    مشکات می‌تواند این تابع را مستقیماً از answer_generator ایمپورت و استفاده کند.
    """
    generator = get_answer_generator()
    async for chunk in generator.get_rag_response_stream(
        query=query,
        chat_id=chat_id,
        filter_doc_ids=filter_doc_ids,
        chat_history=chat_history
    ):
        yield chunk
