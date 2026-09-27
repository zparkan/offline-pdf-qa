"""
ui_schemas.py
----------------------------------
کلاس‌ها و ساختارهای داده‌ای استاندارد برای ارتباط فرانت‌اند (مشکات) با بک‌اند (زینب و فاطمه).
مکمل ui_backend_contract.md.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Dict, Any, Union

# پیام‌های استاندارد وضعیت سیستم
STATUS_COMPLETED = "تکمیل پردازش اسناد"
STATUS_WAITING_PROMPT = "⚠️ در حال پردازش اسناد .... لطفاً منتظر بمانید"
STATUS_NO_DOCS = "⚠️ هنوز هیچ سندی به این گفتگو اضافه نشده است."


@dataclass
class Citation:
    """ساختار اطلاعات ارجاع و منبع پاسخ برای نمایش در UI"""
    doc_id: str
    filename: str
    page_number: int
    score: float
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StreamChunk:
    """
    ساختار قطعات ارسال‌شده در جریان استریم به رابط کاربری.
    - type="token": برای توکن‌های متنی زنده
    - type="sources": برای لیست مراجع در انتهای پاسخ
    """
    type: str  # "token" | "sources" | "error"
    content: Optional[str] = None
    citations: Optional[List[Citation]] = None

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {"type": self.type}
        if self.content is not None:
            res["content"] = self.content
        if self.citations is not None:
            res["citations"] = [c.to_dict() if hasattr(c, "to_dict") else c for c in self.citations]
        return res


# تایپ کال‌بک گزارش وضعیت برای پردازش اسناد
ProgressCallback = Callable[[str, float], None]
