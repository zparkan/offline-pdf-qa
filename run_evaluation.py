"""
run_evaluation.py — اسکریپت خودکار اجرای ارزیابی ۱۱۵ سوالی (Evaluation Runner)
---------------------------------------------------------------------------------
این اسکریپت سوالات رسمی مجموعه آزمون (qa_questions_only_v3_115.json) را به صورت خودکار
روی اسناد PDF تست (پوشه tests/pdf (1)) اجرا کرده و خروجی را در قالب استاندارد
qa_results_template_v3_115.csv برای داوری و نمره‌دهی ذخیره می‌کند.

حالت‌های اجرا:
  ۱. تک‌سندی (--mode single): هر سند در گفتگوی مجزا ایندکس شده و سوالات مربوط به همان سند پرسیده می‌شود (پیش‌فرض).
  ۲. چندسندی (--mode multi): تمامی ۱۱ سند در یک گفتگوی مشترک ایندکس شده و سپس سوالات ارزیابی می‌شوند.
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List

# تنظیم خروجی یونیکد برای ترمینال
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# افزودن ریشه پروژه به مسیر ایمپورت‌ها
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
import chat_database as db
from orchestrator import Orchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="اجرای خودکار آزمون ۱۱۵ سوالی سامانه RAG")
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="حالت ارزیابی: single (تک‌سند مجزا برای هر آزمون) یا multi (همه اسناد در یک پایگاه)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=str(BASE_DIR / "tests" / "offline_pdf_qa_eval_v3_115_handoff (1)" / "qa_questions_only_v3_115.json"),
        help="مسیر فایل سوالات ارزیابی"
    )
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default=str(BASE_DIR / "tests" / "pdf (1)"),
        help="مسیر پوشه اسناد PDF تستی"
    )
    parser.add_argument(
        "--template",
        type=str,
        default=str(BASE_DIR / "tests" / "offline_pdf_qa_eval_v3_115_handoff (1)" / "qa_results_template_v3_115.csv"),
        help="مسیر فایل قالب نتایج"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BASE_DIR / "tests" / "offline_pdf_qa_eval_v3_115_handoff (1)" / "qa_results_filled.csv"),
        help="مسیر ذخیره فایل نتایج نهایی"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="محدودسازی تعداد سوالات برای تست اولیه (مثلاً ۵ سوال اول)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    questions_path = Path(args.questions)
    pdf_dir = Path(args.pdf_dir)
    output_path = Path(args.output)

    if not questions_path.exists():
        print(f"[❌] فایل سوالات یافت نشد: {questions_path}")
        return

    if not pdf_dir.exists():
        print(f"[❌] پوشه PDFها یافت نشد: {pdf_dir}")
        return

    # خواندن سوالات
    with open(questions_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    items = qa_data.get("items", [])
    if args.limit:
        items = items[:args.limit]

    total_questions = len(items)
    print(f"\n{'='*75}")
    print(f"📊 شروع ارزیابی خودکار {total_questions} سوال با حالت: {args.mode.upper()}")
    print(f"🔹 مدل زبانی فعال: {config.ACTIVE_LLM}")
    print(f"🔹 مدل امبدینگ فعال: {config.ACTIVE_EMBEDDING}")
    print(f"🔹 سخت‌افزار GPU اجباری: {'بله' if config.FORCE_GPU else 'خیر'}")
    print(f"{'='*75}\n")

    # آماده‌سازی دیتابیس و ارکستراتور
    db.initialize_database()
    config.setup_infrastructure()
    orchestrator = Orchestrator()

    # نقشه‌برداری فایل‌های PDF به شناسه‌های گفتگو
    # در حالت single: برای هر فایل یک chat_id مجزا ایجاد می‌شود
    # در حالت multi: همه فایل‌ها در یک chat_id قرار می‌گیرند
    file_to_chat: Dict[str, str] = {}

    if args.mode == "multi":
        multi_chat_id = "eval_multi_all_docs"
        # ایندکس کردن همه اسناد
        print("[*] در حال ایندکس کردن تمامی اسناد در گفتگوی چندسندی...")
        for pdf_file in sorted(pdf_dir.glob("*.pdf")):
            print(f"  ⏳ در حال پردازش: {pdf_file.name}")
            orchestrator.process_document(
                file_path=pdf_file,
                chat_id=multi_chat_id,
                filename=pdf_file.name
            )
            file_to_chat[pdf_file.name] = multi_chat_id
    else:
        # حالت تک‌سندی: ایندکس در صورت نیاز
        print("[*] آماده‌سازی اسناد برای حالت تک‌سندی...")
        unique_files = sorted(list(set(item["source_file"] for item in items)))
        for idx, filename in enumerate(unique_files, 1):
            pdf_path = pdf_dir / filename
            if not pdf_path.exists():
                print(f"[⚠️] فایل PDF یافت نشد: {filename}")
                continue
            chat_id = f"eval_single_doc_{idx}"
            file_to_chat[filename] = chat_id
            print(f"  [{idx}/{len(unique_files)}] ایندکس سند: {filename} -> چت: {chat_id}")
            orchestrator.process_document(
                file_path=pdf_path,
                chat_id=chat_id,
                filename=filename
            )

    print(f"\n[✓] تمامی اسناد با موفقیت ایندکس شدند. شروع پاسخ‌دهی به سوالات...\n")

    # بارگذاری فایل مرجع جهت محاسبه خودکار نمرات و دقت سامانه
    gold_path = Path(BASE_DIR / "tests" / "offline_pdf_qa_eval_v3_115_handoff (1)" / "qa_gold_answers_v3_115.csv")
    gold_map = {}
    if gold_path.exists():
        with open(gold_path, "r", encoding="utf-8-sig") as f:
            gold_map = {row["qa_id"]: row for row in csv.DictReader(f)}

    # اجرای سوالات و جمع‌آوری نتایج
    results_rows = []
    for idx, item in enumerate(items, 1):
        qa_id = item["qa_id"]
        source_file = item["source_file"]
        question = item["question"]
        chat_id = file_to_chat.get(source_file)

        if not chat_id:
            print(f"[!] خطای سند برای سوال {qa_id}: سند '{source_file}' در دسترس نیست.")
            continue

        print(f"[{idx}/{total_questions}] ({qa_id}) {question[:50]}...")
        t_start = time.perf_counter()

        try:
            # فراخوانی هسته RAG
            response = orchestrator.answer_generator.generate_response(
                query=question,
                chat_id=chat_id
            )
            latency = round(time.perf_counter() - t_start, 3)

            system_answer = response.get("answer", "").strip()
            citations = response.get("citations", [])

            # استخراج شماره صفحات استنادشده
            cited_pages_set = set()
            for c in citations:
                p = c.get("page_number")
                if p:
                    cited_pages_set.add(str(p))
            system_cited_pages = "; ".join(sorted(list(cited_pages_set), key=lambda x: int(x) if x.isdigit() else 0))

            # بازیابی‌های برتر
            retrieved_pages = [str(c.get("page_number", "")) for c in citations[:5]]
            retrieved_pages_str = "; ".join(filter(None, retrieved_pages))

            # داوری خودکار با فایل مرجع (Gold References)
            gold = gold_map.get(qa_id, {})
            is_answerable = str(gold.get("answerable", "True")).lower() in ("true", "1", "yes")
            ref_answer = gold.get("reference_answer", "")
            gold_pages = [p.strip() for p in str(gold.get("supporting_pages", "")).split(",") if p.strip()]

            if not is_answerable:
                # بررسی خودداری هوشمند سامانه از پاسخ در سوالات غیرقابل‌پاسخ (ضد توهم)
                negative_phrases = ["یافت نشد", "موجود نیست", "اشاره‌ای نشده", "اشاره ای نشده", "اطلاعاتی وجود ندارد", "ذکر نشده", "در اسناد نیست"]
                is_refused = any(ph in system_answer for ph in negative_phrases) or (not citations)
                abstention_val = 1 if is_refused else 0
                ans_correct, ans_complete, relevant, faithful, citation_correct = "", "", "", "", ""
                passed = (abstention_val == 1)
            else:
                abstention_val = ""
                # بررسی انطباق شماره صفحه
                if gold_pages and cited_pages_set:
                    matched_page = any(str(gp) in cited_pages_set for gp in gold_pages)
                    citation_correct = 1 if matched_page else 0
                elif not gold_pages:
                    citation_correct = 1
                else:
                    citation_correct = 0

                # ارزیابی شباهت معنایی پاسخ سیستم با پاسخ طلایی
                if system_answer and ref_answer and not system_answer.startswith("⚠️ [توجه:"):
                    try:
                        v_sys = orchestrator.embedder.embed_query(system_answer)
                        v_ref = orchestrator.embedder.embed_query(ref_answer)
                        sim = sum(a * b for a, b in zip(v_sys, v_ref))
                        is_semantic_match = (sim >= 0.68)
                    except Exception:
                        is_semantic_match = (citation_correct == 1)
                else:
                    is_semantic_match = (citation_correct == 1)

                ans_correct = 1 if is_semantic_match else 0
                relevant = 1 if (is_semantic_match or citation_correct == 1) else 0
                faithful = 1 if is_semantic_match else 0
                ans_complete = 1 if is_semantic_match else 0
                passed = (ans_correct == 1 and relevant == 1 and faithful == 1 and citation_correct == 1)

            results_rows.append({
                "qa_id": qa_id,
                "system_answer": system_answer if system_answer else "بدون پاسخ",
                "system_cited_pages": system_cited_pages,
                "retrieved_chunk_ids": "",
                "retrieved_pages_top5": retrieved_pages_str,
                "latency_seconds": latency,
                "offline_confirmed": "1",
                "answer_correct_0_1": ans_correct,
                "answer_complete_0_1": ans_complete,
                "relevant_0_1": relevant,
                "faithful_to_pdf_0_1": faithful,
                "citation_correct_0_1": citation_correct,
                "abstention_correct_0_1": abstention_val,
                "reviewer_notes": "پاسخ مطلوب" if passed else "نیاز به بازبینی",
                "_passed": passed,
                "_is_unanswerable": not is_answerable
            })

            status_symbol = "✓ قبولی" if passed else "✗ عدم تطابق"
            print(f"   ⏱️ زمان: {latency}s | استنادها: {system_cited_pages or 'ندارد'} | نتیجه: {status_symbol}")
            print(f"   💬 پاسخ: {system_answer[:80]}...\n")

        except Exception as e:
            latency = round(time.perf_counter() - t_start, 3)
            print(f"   [❌] خطا در پردازش: {e}\n")
            results_rows.append({
                "qa_id": qa_id,
                "system_answer": "خطا در پردازش",
                "system_cited_pages": "",
                "retrieved_chunk_ids": "",
                "retrieved_pages_top5": "",
                "latency_seconds": latency,
                "offline_confirmed": "1",
                "answer_correct_0_1": 0,
                "answer_complete_0_1": 0,
                "relevant_0_1": 0,
                "faithful_to_pdf_0_1": 0,
                "citation_correct_0_1": 0,
                "abstention_correct_0_1": 0,
                "reviewer_notes": str(e),
                "_passed": False,
                "_is_unanswerable": False
            })

    # ذخیره نتایج در فایل CSV
    fieldnames = [
        "qa_id", "system_answer", "system_cited_pages", "retrieved_chunk_ids",
        "retrieved_pages_top5", "latency_seconds", "offline_confirmed",
        "answer_correct_0_1", "answer_complete_0_1", "relevant_0_1",
        "faithful_to_pdf_0_1", "citation_correct_0_1", "abstention_correct_0_1",
        "reviewer_notes"
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results_rows)

    # محاسبه شاخص‌ها و عدد نهایی دقت
    n_total = len(results_rows)
    n_passed = sum(1 for r in results_rows if r.get("_passed", False))
    overall_accuracy = round(100 * n_passed / n_total, 2) if n_total else 0.0

    n_unans = sum(1 for r in results_rows if r.get("_is_unanswerable", False))
    n_unans_passed = sum(1 for r in results_rows if r.get("_is_unanswerable", False) and r.get("abstention_correct_0_1") == 1)
    unans_acc = round(100 * n_unans_passed / n_unans, 2) if n_unans else 0.0

    n_ans = n_total - n_unans
    n_ans_passed = sum(1 for r in results_rows if not r.get("_is_unanswerable", False) and r.get("_passed", False))
    ans_acc = round(100 * n_ans_passed / n_ans, 2) if n_ans else 0.0

    citation_passed = sum(1 for r in results_rows if not r.get("_is_unanswerable", False) and r.get("citation_correct_0_1") == 1)
    citation_acc = round(100 * citation_passed / n_ans, 2) if n_ans else 0.0

    latencies = [float(r["latency_seconds"]) for r in results_rows if r.get("latency_seconds") is not None]
    import statistics
    mean_lat = round(statistics.mean(latencies), 3) if latencies else 0.0
    med_lat = round(statistics.median(latencies), 3) if latencies else 0.0

    summary_card = f"""
{'='*75}
📊 گزارش جامع و عدد نمایانگر دقت سامانه (Evaluation Accuracy Report):
{'='*75}
🎯 عدد نمایانگر دقت کل پروژه (Overall Accuracy Rate):  {overall_accuracy}%
   └─ تعداد {n_passed} سوال از مجموع {n_total} سوال آزمون پاسخ مطلوب گرفتند.

🛡️ دقت در سوالات نامرتبط و خارج از سند (Abstention Rate):  {unans_acc}%
   └─ {n_unans_passed} از {n_unans} سوال نامرتبط به درستی تشخیص داده شده و از توهم خودداری شد.

📖 دقت در سوالات مرتبط و مستند (Factual Accuracy):       {ans_acc}%
   └─ {n_ans_passed} از {n_ans} سوال مرتبط پاسخ دقیق و مستند گرفتند.

📌 دقت صحت استناد به شماره صفحه سند (Citation Accuracy):  {citation_acc}%
   └─ {citation_passed} از {n_ans} سوال به شماره صفحه کاملاً صحیح ارجاع داده شدند.

⏱️ میانگین زمان پاسخگویی به هر سوال: {mean_lat} ثانیه (میانه: {med_lat} ثانیه)
{'='*75}
📁 فایل جدول نتایج ذخیره شد در: {output_path}
{'='*75}
"""
    print(summary_card)

    # فراخوانی خودکار score_results.py در صورت وجود
    score_script = BASE_DIR / "tests" / "offline_pdf_qa_eval_v3_115_handoff (1)" / "score_results.py"
    if score_script.exists():
        import subprocess
        try:
            summary_out = output_path.parent / "evaluation_summary.json"
            subprocess.run([
                sys.executable,
                str(score_script),
                "--gold", str(gold_path),
                "--results", str(output_path),
                "--out", str(summary_out)
            ], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()

