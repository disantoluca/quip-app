import faiss, pickle
from pathlib import Path

INDEX = Path("data/index/index.faiss")
META = Path("data/index/meta.pkl")

if not INDEX.exists():
    print("❌ No FAISS index found at", INDEX)
    exit(1)

idx = faiss.read_index(str(INDEX))
print(f"✅ FAISS index loaded — {idx.ntotal} vectors")

if META.exists():
    meta = pickle.load(open(META, "rb"))
    docs = meta.get("docs") or meta.get("texts") or []
    print(f"🗂️  Metadata contains {len(docs)} documents")
    if docs:
        print("🔹 Sample snippet:")
        print(docs[0][:300].replace("\n", " "))
else:
    print("⚠️ No metadata file found at", META)