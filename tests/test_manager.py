print("[1] در حال شروع اسکریپت...")
try:
    from model_manager import ModelManager
    import os
    print("[2] کتابخانه‌ها با موفقیت لود شدند.")
except Exception as e:
    print(f"[X] خطا در لود کردن کتابخانه‌ها: {e}")
    exit()

def run_test():
    print("[3] وارد تابع تست شدیم.")
    manager = ModelManager(root_dir="test_models_folder")
    
    print("[4] در حال چک کردن پوشه‌ها...")
    if os.path.exists("test_models_folder/embedding"):
        print("[✓] پوشه‌ها ساخته شدند.")
    
    print("[5] در حال تست دانلود (لطفاً کمی صبر کنید، ممکنه طول بکشه)...")
    status, msg, path = manager.get_embedding_model(
        repo_id="hf-internal-testing/tiny-random-BertModel", 
        model_name="tiny-test"
    )
    print(f"[6] نتیجه تست امبدینگ: {msg}")

if __name__ == "__main__":
    run_test()