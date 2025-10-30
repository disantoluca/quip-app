#!/usr/bin/env python3
"""
Rebuild FAISS index from documents in data/persist/
Usage: python rebuild_index.py
"""

import os
import pickle
import faiss
from pathlib import Path

# Try to import sentence-transformers, fall back to sklearn if not available
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False

def load_documents():
    """Load all documents from data/persist/ or existing corpus"""
    docs = []

    # Try to load from data/persist/ first
    persist_dir = Path("data/persist")
    if persist_dir.exists():
        for file_path in persist_dir.glob("*.pkl"):
            try:
                with open(file_path, 'rb') as f:
                    doc_data = pickle.load(f)
                    docs.extend(doc_data if isinstance(doc_data, list) else [doc_data])
                    print(f"✅ Loaded {file_path.name}")
            except Exception as e:
                print(f"⚠️ Failed to load {file_path.name}: {e}")

    # Try to load from existing corpus file
    if not docs:
        corpus_file = Path("corpus_data.pkl")
        if corpus_file.exists():
            try:
                with open(corpus_file, 'rb') as f:
                    corpus_data = pickle.load(f)
                    docs = corpus_data
                    print(f"✅ Loaded {len(docs)} docs from {corpus_file}")
            except Exception as e:
                print(f"⚠️ Failed to load {corpus_file}: {e}")

    if not docs:
        print("❌ No data/persist/ directory found and no corpus_data.pkl")
        print("💡 Try running the Streamlit app first to create some documents")
        print("💡 Or manually create data/persist/ and add .pkl files there")

    return docs

def rebuild_index(docs):
    """Rebuild FAISS index from documents"""
    if not docs:
        print("❌ No documents to index")
        return False

    print(f"🔄 Processing {len(docs)} documents...")

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

    # Generate embeddings based on available libraries
    if HAS_SENTENCE_TRANSFORMERS:
        print("🧠 Generating embeddings with sentence-transformers...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=True)

        # Create FAISS index
        print("🔍 Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

    elif HAS_SKLEARN:
        print("🧠 Generating TF-IDF vectors with sklearn...")
        vectorizer = TfidfVectorizer(max_features=384, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        embeddings = tfidf_matrix.toarray().astype('float32')

        # Create FAISS index
        print("🔍 Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

    else:
        print("❌ No embedding library available!")
        print("💡 Install sentence-transformers: conda install -c conda-forge sentence-transformers")
        print("💡 Or install sklearn: conda install scikit-learn")
        return False

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