"""
model_manager.py — ماژول مدیریت، آماده‌سازی و دانلود خودکار مدل‌های هوش مصنوعی
---------------------------------------------------------------------------------
وظایف:
  ۱. بررسی و ساخت خودکار پوشه ریشه models و زیرپوشه‌های embedding و llm در هر بار اجرا.
  ۲. خواندن مدل‌های فعال از config و بررسی وجود فایل‌های آن‌ها روی دیسک محلی.
  ۳. دانلود هوشمند و بدون اختلال فایل‌های حجیم GGUF مدل زبانی (با Progress Bar زنده، قابلیت Resume و سرور Mirror).
  ۴. عدم نیاز به هیچ اسکریپت دانلود دستی یا اختصاصی برای مدل‌های مختلف.
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class ModelManager:
    """
    مدیریت متمرکز و یکپارچه پوشه‌ها و فایل‌های مدل‌های زبانی و امبدینگ.
    """

    def __init__(self, root_dir: str = "models"):
        self.root_dir = Path(root_dir)
        self.embedding_dir = self.root_dir / "embedding"
        self.llm_dir = self.root_dir / "llm"
        self._create_directories()

    def _create_directories(self) -> None:
        """بررسی و ایجاد پوشه‌های اصلی مدل‌ها در صورت عدم وجود."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.llm_dir.mkdir(parents=True, exist_ok=True)

    def get_embedding_model(self, repo_id: str, model_name: str) -> Tuple[bool, str, Optional[str]]:
        """
        بررسی وجود یا دانلود مدل امبدینگ از Hugging Face.
        """
        target_path = self.embedding_dir / model_name
        if target_path.exists() and any(target_path.iterdir()):
            return True, f"EXIST: {model_name}", str(target_path)

        try:
            print(f"[*] در حال دریافت مدل امبدینگ '{model_name}' از ریپازیتوری '{repo_id}'...")
            model = SentenceTransformer(repo_id)
            model.save(str(target_path))
            print(f"[✓] مدل امبدینگ '{model_name}' با موفقیت ذخیره شد.")
            return True, f"DOWNLOADED: {model_name}", str(target_path)
        except Exception as e:
            err_msg = f"خطا در دریافت مدل امبدینگ: {e}"
            print(f"[❌] {err_msg}")
            return False, err_msg, None

    def _download_gguf_file(
        self,
        repo_id: str,
        filename: str,
        target_dir: Path
    ) -> Tuple[bool, str, Optional[str]]:
        """
        دانلود هوشمند فایل‌های GGUF با نمایش درصد پیشرفت زنده، قابلیت Resume و سرور پشتیبان Mirror.
        """
        target_file = target_dir / filename
        part_file = target_dir / f"{filename}.part"

        # اگر فایل کامل و سالم از قبل وجود دارد، دانلود مجدد انجام نشود
        if target_file.exists() and target_file.stat().st_size > 1024 * 1024:
            return True, f"LOCAL READY: {filename}", str(target_file)

        # پاکسازی هرگونه فایل قفل بجا مانده از پروسه‌های قبلی
        lock_pattern = f"{filename}*.lock"
        for lock_f in target_dir.glob(lock_pattern):
            try:
                lock_f.unlink()
            except OSError:
                pass

        # آدرس‌های دانلود (سرور رسمی هاگینگ‌فیس و سرور آینه‌ای کمکی)
        urls = [
            f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
            f"https://hf-mirror.com/{repo_id}/resolve/main/{filename}"
        ]

        print(f"\n{'='*75}")
        print(f"📥 مدل زبانی فعال در پوشه محلی یافت نشد.")
        print(f"🔹 نام فایل: {filename}")
        print(f"🔹 مخزن: {repo_id}")
        print(f"🔹 مسیر ذخیره‌سازی: {target_dir}")
        print(f"[*] در حال اتصال جهت دانلود خودکار مدل...")
        print(f"{'='*75}\n")

        last_error = None
        for candidate_url in urls:
            try:
                existing_size = part_file.stat().st_size if part_file.exists() else 0
                headers = {"User-Agent": "Mozilla/5.0"}
                if existing_size > 0:
                    headers["Range"] = f"bytes={existing_size}-"
                    print(f"[*] ادامه دانلود فایل از بایت {existing_size:,} (حالت Resume)...")

                response = requests.get(
                    candidate_url,
                    headers=headers,
                    stream=True,
                    allow_redirects=True,
                    timeout=30
                )

                if response.status_code == 416:
                    # در صورت نامعتبر بودن محدوده، از ابتدا شروع می‌کنیم
                    existing_size = 0
                    headers.pop("Range", None)
                    response = requests.get(
                        candidate_url,
                        headers=headers,
                        stream=True,
                        allow_redirects=True,
                        timeout=30
                    )

                if response.status_code not in (200, 206):
                    response.raise_for_status()

                content_len = response.headers.get("content-length")
                if content_len is not None:
                    total_bytes = int(content_len) + (existing_size if response.status_code == 206 else 0)
                else:
                    total_bytes = None

                mode = "ab" if (response.status_code == 206 and existing_size > 0) else "wb"
                initial_bytes = existing_size if response.status_code == 206 else 0

                # دانلود چانک به چانک با نوار پیشرفت زنده در کنسول
                chunk_size = 2 * 1024 * 1024  # ۲ مگابایت در هر چانک
                with open(part_file, mode) as f_out, tqdm(
                    desc=f"دریافت {filename[:28]}",
                    total=total_bytes,
                    initial=initial_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    ncols=85
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f_out.write(chunk)
                            pbar.update(len(chunk))

                # پس از اتمام موفق، فایل موقت به فایل اصلی تغییر نام داده می‌شود
                if part_file.exists():
                    if target_file.exists():
                        target_file.unlink()
                    part_file.rename(target_file)

                size_mb = target_file.stat().st_size / (1024 * 1024)
                print(f"\n[✓] فایل مدل زبانی با موفقیت ذخیره شد ({size_mb:.1f} مگابایت).")
                print(f"[✓] آدرس فایل نهایی: {target_file}\n")
                return True, f"DOWNLOADED: {filename}", str(target_file)

            except Exception as ex:
                last_error = ex
                print(f"[!] ارتباط با سرور ({candidate_url.split('/')[2]}) ناموفق بود: {ex}")
                print("[*] در حال تلاش برای بررسی آدرس بعدی...")
                continue

        # در صورت بروز مشکل با دانلود مستقیم، تلاش نهایی از طریق API کتابخانه huggingface_hub
        try:
            print("[*] در حال تلاش از طریق کتابخانه huggingface_hub...")
            from huggingface_hub import hf_hub_download
            res_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False
            )
            return True, f"DOWNLOADED: {filename}", str(res_path)
        except Exception as e:
            msg = f"خطا در دانلود خودکار مدل زبانی: {last_error or e}"
            print(f"[❌] {msg}")
            return False, msg, None

    def setup_llm(
        self,
        mode: str,
        model_name_or_repo: str,
        filename: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        راه‌اندازی و اطمینان از وجود فایل مدل زبانی در حالت local یا ollama.
        """
        if mode == "ollama":
            import subprocess
            try:
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
                if model_name_or_repo in result.stdout:
                    return True, f"OLLAMA READY: {model_name_or_repo}", "ollama"
                else:
                    return False, f"OLLAMA MODEL NOT FOUND: {model_name_or_repo}", None
            except Exception:
                return False, "OLLAMA NOT INSTALLED", None

        elif mode == "local":
            if not filename:
                return False, "نام فایل GGUF برای حالت محلی مشخص نشده است.", None
            return self._download_gguf_file(
                repo_id=model_name_or_repo,
                filename=filename,
                target_dir=self.llm_dir
            )
        else:
            return False, f"حالت نامعتبر: {mode}", None

    def ensure_active_models(self, auto_download_llm: Optional[bool] = None) -> Dict[str, Any]:
        """
        بررسی خودکار و جامع مدل‌های فعال پروژه بر اساس config.py.
        اگر فایل مدل موجود نباشد و دانلود خودکار غیرفعال باشد، سیستم به صورت آزمایشی (Mock Mode) بالا می‌آید
        تا کاربر بدون گیر کردن در دانلود و فیلترینگ اینترنت بتواند برنامه را تست کند.
        """
        import config
        status_report = {}

        # ۱. بررسی مدل امبدینگ فعال
        emb_key = config.ACTIVE_EMBEDDING
        emb_info = config.EMBEDDING_MODELS.get(emb_key, {})
        if emb_info:
            success, msg, path = self.get_embedding_model(
                repo_id=emb_info["repo_id"],
                model_name=emb_key
            )
            status_report["embedding"] = {
                "success": success,
                "key": emb_key,
                "msg": msg,
                "path": path
            }
            print(f"[*] وضعیت مدل امبدینگ فعال ({emb_key}): {msg}")

        # ۲. بررسی مدل زبانی فعال
        llm_key = config.ACTIVE_LLM
        llm_info = config.LLM_MODELS.get(llm_key, {})
        should_download = auto_download_llm if auto_download_llm is not None else getattr(config, "AUTO_DOWNLOAD_LLM", True)

        if llm_info:
            target_file = self.llm_dir / llm_info.get("filename", "")
            if target_file.exists() and target_file.stat().st_size > 1024 * 1024:
                status_report["llm"] = {
                    "success": True,
                    "key": llm_key,
                    "msg": f"LOCAL READY: {target_file.name}",
                    "path": str(target_file)
                }
                print(f"[✓] مدل زبانی محلی در دیسک آماده است: {target_file.name}")
            elif not should_download:
                status_report["llm"] = {
                    "success": False,
                    "key": llm_key,
                    "msg": "فایل مدل هنوز در دیسک نیست (سیستم در حالت شبیه‌ساز Mock اجرا می‌شود)",
                    "path": None
                }
                print(f"[*] فایل مدل زبانی '{llm_key}' یافت نشد. سیستم در حالت شبیه‌ساز (Mock Mode) اجرا می‌شود.")
            else:
                success, msg, path = self.setup_llm(
                    mode="local",
                    model_name_or_repo=llm_info["repo"],
                    filename=llm_info.get("filename")
                )
                status_report["llm"] = {
                    "success": success,
                    "key": llm_key,
                    "msg": msg,
                    "path": path
                }
                print(f"[*] وضعیت مدل زبانی فعال ({llm_key}): {msg}")

        return status_report
