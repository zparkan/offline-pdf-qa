# reset_db.py
# هشدار: این اسکریپت تمام داده‌های ChromaDB را پاک می‌کند
# فقط هنگام تغییر مدل امبدینگ اجرا کنید

import chromadb
from config import CHROMA_DB_PATH, COLLECTION_NAME

def reset_all_collections():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    existing = client.list_collections()
    for col in existing:
        client.delete_collection(col.name)
        print(f"حذف شد: {col.name}")
    
    print("DB پاک شد. حالا pipeline را دوباره اجرا کنید.")

if __name__ == "__main__":
    confirm = input("مطمئنی؟ همه داده‌ها حذف می‌شن (y/n): ")
    if confirm.lower() == "y":
        reset_all_collections()
