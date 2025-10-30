#!/usr/bin/env python3
"""
Rebuild FAISS index from documents in data/persist/
Usage: python rebuild_index.py
"""

import os
import pickle
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

def load_documents():
    """Load all documents from data/persist/"""
    persist_dir = Path("data/persist")
    if not persist_dir.exists():
        print("❌ No data/persist/ directory found")
        return []

    docs = []
    for file_path in persist_dir.glob("*.pkl"):
        try:
            with open(file_path, 'rb') as f:
                doc_data = pickle.load(f)
                docs.extend(doc_data if isinstance(doc_data, list) else [doc_data])
                print(f"✅ Loaded {file_path.name}")
        except Exception as e:
            print(f"⚠️ Failed to load {file_path.name}: {e}")

    return docs

def rebuild_index(docs):
    """Rebuild FAISS index from documents"""
    if not docs:
        print("❌ No documents to index")
        return False

    print(f"🔄 Processing {len(docs)} documents...")

    # Initialize sentence transformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract texts and metadata
    texts = []
    metas = []

    for doc in docs:
        text = doc.get("text", "")
        if text.strip():
            # Split into chunks if needed
            chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
            for chunk in chunks:
                if chunk.strip():
                    texts.append(chunk)
                    metas.append(doc.get("meta", {}))

    if not texts:
        print("❌ No valid text content found")
        return False

    print(f"📝 Created {len(texts)} text chunks")

    # Generate embeddings
    print("🧠 Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Create FAISS index
    print("🔍 Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save index and metadata
    index_dir = Path("data/index")
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "index.faiss"))

    with open(index_dir / "meta.pkl", 'wb') as f:
        pickle.dump({"docs": texts, "metas": metas}, f)

    print(f"✅ Index saved with {index.ntotal} vectors")
    print(f"📁 Files: {index_dir}/index.faiss, {index_dir}/meta.pkl")

    return True

def main():
    print("🚀 Rebuilding FAISS index...")

    # Load documents
    docs = load_documents()

    # Rebuild index
    success = rebuild_index(docs)

    if success:
        print("\n🎉 Index rebuild complete!")
        print("💡 Now restart your Streamlit app to use the new index")
    else:
        print("\n❌ Index rebuild failed")

if __name__ == "__main__":
    main()