# streamlit_app.py
import os
import re
import json
import time
import base64
import requests
import zipfile
import openai
import faiss
import pickle
import numpy as np
from urllib.parse import urlparse
import streamlit as st
import pandas as pd
import io
from io import StringIO
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
from openai import OpenAI
client =OpenAI()
# Optional PDF parsing
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

# Optional ML deps (we fall back gracefully)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

APP_TITLE = "CAP Country Folder Q&A — Quip + Local Docs"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Upload Quip exports or fetch via API, index the content, and ask questions interactively.")

# -------------------------
# Utilities
# -------------------------
# --- Utility functions ---

def html_to_text(html: str) -> str:
    """Convert Quip HTML content into plain readable text."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200):
    """
    Split large text into overlapping chunks for embedding or search indexing.
    Example: chunk_text(long_doc, 1500, 200) returns list of overlapping chunks.
    """
    if not text:
        return []

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def first_existing(d: dict, *keys):
    """
    Utility to get the first existing key from a dict, e.g. first_existing(doc, "html", "rendered_html")
    """
    for k in keys:
        if d.get(k):
            return d[k]
    return None
def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def pdf_to_text(file) -> str:
    if not HAS_PYPDF:
        return ""
    reader = PdfReader(file)
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            pass
    return clean_text("\n".join(chunks))

def fetch_all_threads_in_folder(folder_id: str, base_url: str, headers: dict) -> list:
    """
    Recursively collect all thread_ids from a folder and its subfolders.
    """
    all_threads = []
    folder_url = f"{base_url}/1/folders/{folder_id}"
    fr = requests.get(folder_url, headers=headers, timeout=30)
    if fr.status_code != 200:
        st.warning(f"⚠️ Folder {folder_id}: {fr.status_code}")
        return all_threads

    folder_data = fr.json().get("folder", {})
    children = folder_data.get("children", [])

    for child in children:
        ctype = child.get("type")
        if ctype == "thread" and child.get("thread_id"):
            all_threads.append(child["thread_id"])
        elif ctype == "folder" and child.get("folder_id"):
            # recursively explore subfolder
            sub_id = child["folder_id"]
            st.write(f"↳ Diving into subfolder {sub_id}")
            all_threads.extend(fetch_all_threads_in_folder(sub_id, base_url, headers))

    return all_threads

def fetch_quip_threads_single(thread_ids_or_tokens: str, token: str, base_url: str) -> dict:
    """
    Resolves each supplied token (public link or internal id) and fetches
    full thread data directly via /1/threads/<id>.
    Returns a dict shaped like {'threads': {id: thread_obj}} for compatibility.
    """
    import requests
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token.strip()}"}

    # --- resolve public tokens to internal ids ---
    ids = [i.strip() for i in thread_ids_or_tokens.strip().splitlines() if i.strip()]
    resolved = {}
    for token_or_id in ids:
        meta_url = f"{base}/1/threads/{token_or_id}"
        r = requests.get(meta_url, headers=headers, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Failed metadata lookup for {token_or_id}: {r.text[:200]}")
        meta = r.json().get("thread")
        thread_id = meta.get("id", token_or_id)

        # --- fetch full thread (same endpoint) ---
        full_url = f"{base}/1/threads/{thread_id}?include_html=true"
        r2 = requests.get(full_url, headers=headers, timeout=30)
        if r2.status_code != 200:
            raise RuntimeError(f"Failed fetch for {thread_id}: {r2.text[:200]}")

        doc = r2.json().get("thread", {})
        resolved[thread_id] = doc

    return {"threads": resolved}


def extract_quip_document_text(thread_json: Dict) -> Tuple[str, Dict]:
    """
    Extracts the main HTML from the Quip thread and converts to text.
    Returns (text, metadata)
    """
    doc_map = thread_json.get("threads", {})
    if not doc_map:
        return "", {}
    # Grab the first (or only) doc
    first_key = next(iter(doc_map.keys()))
    doc = doc_map[first_key]
    html = doc.get("html", "")
    text = html_to_text(html)
    meta = {
        "id": doc.get("id"),
        "title": doc.get("title"),
        "updated_usec": doc.get("updated_usec"),
        "created_usec": doc.get("created_usec"),
    }
    return text, meta

def ensure_state():
    if "docs" not in st.session_state:
        st.session_state.docs = []  # list of dicts: {source, text, meta}
    if "chunks" not in st.session_state:
        st.session_state.chunks = []  # list of dicts: {source, text, meta, chunk_id}
    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
    if "matrix" not in st.session_state:
        st.session_state.matrix = None

ensure_state()

TOKEN_RE = re.compile(r"https?://[^/]*quip[^/]+/([A-Za-z0-9]{8,})")

def extract_quip_tokens_from_html(html: str) -> list:
    """Extract all Quip tokens from hyperlinks inside a document HTML."""
    if not html:
        return []
    return list(dict.fromkeys(TOKEN_RE.findall(html)))  # dedupe while preserving order

def fetch_thread_doc(token_or_id: str, base_url: str, headers: dict) -> dict:
    """Fetch a single thread by token or ID (works with standard user token)."""
    import requests
    url = f"{base_url.rstrip('/')}/1/threads/{token_or_id}?include_html=true"
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        return {}
    return r.json().get("thread", {})
def _docs_signature():
    """Stable signature of the current corpus; changes when docs change."""
    import hashlib, json
    docs = st.session_state.get("docs", [])
    payload = [
        {
            "id": (d.get("meta") or {}).get("id"),
            "updated": (d.get("meta") or {}).get("updated_usec"),
            "len": len(d.get("text","")),
            "source": d.get("source"),
        }
        for d in docs
    ]
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(raw).hexdigest()

def crawl_quip_links(seed_items: list, base_url: str, headers: dict, max_pages: int = 50) -> dict:
    """BFS crawl of linked Quip documents starting from one or more seeds."""
    seen = set()
    queue = []
    results = {}

    # Normalize input (accept full URLs or bare tokens)
    for item in seed_items:
        item = item.strip()
        if not item:
            continue
        if item.startswith("http"):
            m = TOKEN_RE.search(item)
            if m:
                queue.append(m.group(1))
        else:
            queue.append(item)

    # Crawl breadth-first up to max_pages
    while queue and len(results) < max_pages:
        tok = queue.pop(0)
        if tok in seen:
            continue
        seen.add(tok)

        doc = fetch_thread_doc(tok, base_url, headers)
        if not doc:
            continue

        tid = doc.get("id") or tok
        results[tid] = doc

        html = doc.get("html") or doc.get("rendered_html") or doc.get("document_html") or ""
        for t in extract_quip_tokens_from_html(html):
            if t not in seen:
                queue.append(t)

    return results


# ✅ Embedding backend for OpenAI or SentenceTransformers
class EmbeddingBackend:
    """Wrapper to handle both OpenAI and SentenceTransformers embedding backends."""
    def __init__(self):
        self.backend = None
        self.model = None
        self.dim = None

    def init(self):
        import os
        # Prefer OpenAI API if key exists
        if os.getenv("OPENAI_API_KEY"):
            self.backend = "openai"
            self.model = "text-embedding-3-small"  # 1536-dim
            self.dim = 1536
            return self

        # Otherwise fallback to SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            self.backend = "sbert"
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.dim = 384
            return self
        except Exception:
            raise RuntimeError(
                "No embedding backend available. Set OPENAI_API_KEY or install sentence-transformers."
            )

    def encode(self, texts):
        """Encode list of texts into embeddings depending on backend."""
        import numpy as np
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self.backend == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            vecs = []
            for chunk in texts:
                resp = client.embeddings.create(
                    model=self.model,
                    input=chunk
                )
                vecs.append(resp.data[0].embedding)
            return np.array(vecs, dtype=np.float32)
        else:
            # SentenceTransformers backend
            return self.model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            ).astype(np.float32)
# 🔹 Top-level helper (NOT inside the class)
def _embedder():
    """Singleton to initialize or reuse embedding backend across Streamlit session."""
    if "embedder" not in st.session_state:
        st.session_state.embedder = EmbeddingBackend().init()
    return st.session_state.embedder

# ---------------- FAISS index build & search ----------------
def _ensure_index():
    """Build an in-memory FAISS index from st.session_state.docs (with chunking)."""
    import numpy as np, faiss

    if not st.session_state.get("docs"):
        st.warning("No documents in corpus yet. Ingest or crawl first.")
        return False

    embedder = _embedder()
    # you can wire these to sidebar sliders later
    chunk_size = 1500
    overlap    = 200

    texts, metas = [], []
    for d in st.session_state.docs:
        txt = d.get("text","")
        if not txt:
            continue
        parts = chunk_text(txt, chunk_size=chunk_size, overlap=overlap)
        for i, p in enumerate(parts):
            texts.append(p)
            m = {**(d.get("meta") or {}), "source": d.get("source"), "chunk_id": i}
            metas.append(m)

    if not texts:
        st.error("No text available to index.")
        return False

    # Embed and normalize for cosine via inner product
    vecs = embedder.encode(texts).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    st.session_state.faiss_index = index
    st.session_state.faiss_texts = texts
    st.session_state.faiss_meta  = metas
    st.success(f"Indexed {len(texts)} chunks from {len(st.session_state.docs)} document(s).")
    return True


def _search(query: str, top_k: int = 5):
    """Search current FAISS index and return ranked passages with metadata."""
    import numpy as np
    if "faiss_index" not in st.session_state:
        st.warning("No index in memory. Click 'Create / Rebuild Index' first.")
        return []

    embedder = _embedder()
    qv = embedder.encode([query])[0].astype("float32")
    qv /= (np.linalg.norm(qv) + 1e-12)

    D, I = st.session_state.faiss_index.search(qv[np.newaxis, :], top_k)
    hits = []
    for rank, idx in enumerate(I[0].tolist(), 1):
        if idx == -1:
            continue
        hits.append({
            "rank": rank,
            "text": st.session_state.faiss_texts[idx],
            "meta": st.session_state.faiss_meta[idx],
            "score": float(D[0][rank-1]),
        })
    return hits
# -------------------------
# -------------------------
# Sidebar: Ingestion panel
# -------------------------
# --------------------------------------------------
# ⚡ Auto-rebuild toggle (sidebar)
# --------------------------------------------------
with st.sidebar:
    st.header("Ingest Documents")
    st.write("Add content from Quip API or upload local files (TXT, HTML, PDF).")

    st.checkbox(
        "⚡ Auto-rebuild index when corpus changes",
        value=True,
        key="auto_index",
        help="If on, FAISS is rebuilt automatically whenever documents are added/updated."
    )

    # ----------- Quip API block -----------
    with st.expander("Fetch from Quip API", expanded=False):
        quip_base = st.text_input(
            "Quip Platform Base URL",
            value="https://platform.quip-apple.com",
            help="Enterprise instances may use https://platform.quip-apple.com or similar."
        )
        quip_token = st.text_input("Quip API Token", type="password")
        thread_ids = st.text_area("Thread IDs (one per line)", placeholder="yEZuAgiIXp8V\n...")

        colA, colB = st.columns(2)
        with colA:
            if st.button("Validate Token", key="validate_token_btn"):
                try:
                    url = f"{quip_base.rstrip('/')}/1/users/current"
                    resp = requests.get(
                        url,
                        headers={"Authorization": f"Bearer {quip_token.strip()}"},
                        timeout=20,
                    )
                    if resp.status_code == 200:
                        me = resp.json()
                        st.success(f"Token OK: {me.get('user', {}).get('name', 'Unknown user')}")
                    else:
                        st.error(f"Token check failed: {resp.status_code} {resp.text[:200]}")
                except Exception as e:
                    st.error(f"Token check error: {e}")

        with colB:
            if st.button("Fetch Threads", key="fetch_threads_btn"):
                if not quip_token or not thread_ids.strip():
                    st.error("Provide API token and at least one thread ID.")
                else:
                    ids = [i.strip() for i in thread_ids.strip().splitlines() if i.strip()]
                    try:
                        # 1) Fetch via per-ID approach
                        data = fetch_quip_threads_single("\n".join(ids), quip_token, quip_base)

                        # 2) Debug: raw API response
                        with st.expander("🔍 Debug: Raw Quip API Response", expanded=False):
                            st.json(data)

                        # 3) Normalize shape
                        threads = data.get("threads") or {}
                        if not threads and "thread" in data:
                            t = data["thread"]
                            threads = {t.get("id", "unknown"): t}

                        if not threads:
                            st.warning("No threads returned. Check IDs and access.")
                            st.stop()

                        # Helper (or use your global one)
                        def first_existing(d: dict, *keys):
                            for k in keys:
                                if d.get(k):
                                    return d[k]
                            return None

                        # 4) Extract, append, and report
                        num_added = 0
                        for tid, doc in threads.items():
                            html = first_existing(
                                doc,
                                "html", "rendered_html", "document_html", "content_html",
                                "body_html", "full_html"
                            )
                            text = html_to_text(html) if html else (doc.get("title", "") or "")

                            meta = {
                                "id": doc.get("id") or tid,
                                "title": doc.get("title"),
                                "updated_usec": doc.get("updated_usec"),
                                "created_usec": doc.get("created_usec"),
                                "link": doc.get("link"),
                                "_has_html": bool(html),
                                "_available_keys": sorted(list(doc.keys())),
                            }

                            st.session_state.docs.append({
                                "source": f"quip:{tid}",
                                "text": text,
                                "meta": meta
                            })
                            num_added += 1

                        st.success(f"Added {num_added} thread(s) to the corpus.")

                    except Exception as e:
                        st.error(f"Error fetching threads: {e}")
                        st.stop()

    # --- Crawl linked Quip docs ---
    with st.expander("🕷️ Crawl linked Quip docs", expanded=False):
        seed_input = st.text_area(
            "Seed Quip doc URLs or tokens (one per line)",
            key="crawl_seed",
            placeholder="https://quip-apple.com/yEZuAgiIXp8V\nyEZuAgiIXp8V\n...",
            help="Paste CAP index/reference *document* URLs or tokens — the crawler follows Quip links and ingests linked docs."
        )
        max_pages = st.number_input(
            "Max pages to crawl",
            min_value=1, max_value=500, value=50, step=10,
            key="crawl_max"
        )

        if st.button("Crawl & Ingest", key="crawl_btn"):
            if not quip_token or not seed_input.strip():
                st.error("Provide API token and at least one seed URL/token.")
            else:
                headers = {"Authorization": f"Bearer {quip_token.strip()}"}
                seeds = [s for s in seed_input.splitlines() if s.strip()]
                try:
                    found = crawl_quip_links(seeds, quip_base, headers, max_pages=int(max_pages))

                    def first_existing(d: dict, *keys):
                        for k in keys:
                            if d.get(k):
                                return d[k]
                        return None

                    num_added = 0
                    for tid, doc in found.items():
                        html = first_existing(
                            doc,
                            "html", "rendered_html", "document_html",
                            "content_html", "body_html", "full_html"
                        )
                        text = html_to_text(html) if html else (doc.get("title", "") or "")
                        meta = {
                            "id": doc.get("id") or tid,
                            "title": doc.get("title"),
                            "link": doc.get("link"),
                            "updated_usec": doc.get("updated_usec"),
                            "created_usec": doc.get("created_usec"),
                            "_has_html": bool(html),
                        }
                        st.session_state.docs.append({
                            "source": f"quip:{tid}",
                            "text": text,
                            "meta": meta
                        })
                        num_added += 1

                    st.success(f"Crawled {len(found)} doc(s). Added {num_added} to the corpus.")
                    with st.expander("🔍 Debug: Crawl Results (IDs & Titles)", expanded=False):
                        st.json({k: found[k].get("title") for k in found})

                except Exception as e:
                    st.error(f"Crawl error: {e}")

    ## # 🧪 Mock data (for testing)
with st.expander("🧪 Mock data (for testing)", expanded=False):
    if st.button("Add Montenegro mock doc"):
        if "docs" not in st.session_state:
            st.session_state.docs = []
        st.session_state.docs.append({
            "source": "mock:montenegro",
            "text": (
                "Driving rules in Montenegro: "
                "U-turns are generally prohibited in areas marked with solid lines "
                "or near intersections. However, they are permitted when visibility "
                "and traffic conditions allow, unless a sign explicitly forbids it. "
                "Always yield to oncoming traffic before making a U-turn."
            ),
            "meta": {
                "id": "mock001",
                "title": "Montenegro Driving Rules",
                "link": "https://example.com/montenegro-driving",
                "created_usec": "20250101000000",
                "updated_usec": "20250102000000"
            }
        })
        st.success("✅ Mock document added. Go to ❓ Ask Questions and click 'Create / Rebuild Index'.")

# --------------------------------------------------
# Auto index maintenance (AFTER sidebar, BEFORE tabs)
# --------------------------------------------------
if "last_index_sig" not in st.session_state:
    st.session_state.last_index_sig = None

try:
    current_sig = _docs_signature()
    if st.session_state.get("auto_index", False) and current_sig:
        if st.session_state.last_index_sig != current_sig:
            if _ensure_index():
                st.session_state.last_index_sig = current_sig
                st.toast("Index rebuilt for latest corpus ✅", icon="✅")
except Exception as e:
    st.warning(f"Auto-index check failed: {e}")

# Main area: Browse & QA
# -------------------------
tab1, tab2 = st.tabs(["📚 Corpus", "❓ Ask Questions"])

with tab1:
    st.subheader("Ingested Documents")

    # -------- Corpus Filter --------
    with st.container():
        colq, colopt1, colopt2 = st.columns([3, 1, 1])
        with colq:
            corpus_query = st.text_input(
                "Filter documents (title & text)",
                placeholder="e.g., Montenegro, lane guidance, Serbia",
                key="corpus_query",
            )
        with colopt1:
            case_sensitive = st.checkbox("Case sensitive", value=False, key="corpus_case")
        with colopt2:
            in_titles_only = st.checkbox("Titles only", value=False, key="corpus_titles_only")

        def _match(hay: str, needle: str) -> bool:
            if not needle:
                return True
            if not case_sensitive:
                hay, needle = hay.lower(), needle.lower()
            return needle in hay

        # Build filtered list (don’t mutate session_state.docs)
        if "docs" not in st.session_state:
            st.session_state.docs = []

        filtered_docs = []
        for d in st.session_state.docs:
            meta = d.get("meta", {})
            title = meta.get("title", "")
            if in_titles_only:
                ok = _match(title, corpus_query)
            else:
                text = d.get("text", "")
                ok = _match(title, corpus_query) or _match(text, corpus_query)
            if ok:
                filtered_docs.append(d)

        st.caption(f"Showing {len(filtered_docs)} of {len(st.session_state.docs)} documents")

        # -------- Document display --------
        if not st.session_state.docs:
            st.info("No documents yet. Add some from the sidebar.")
        elif not filtered_docs:
            st.warning("No matches for your filter.")
        else:
            for i, d in enumerate(filtered_docs, 1):
                meta = d.get("meta", {})
                title = meta.get("title", f"Document {i}")
                link = meta.get("link")
                created = meta.get("created_usec")
                updated = meta.get("updated_usec")

                with st.expander(f"{i}. {title}", expanded=False):
                    if link:
                        st.markdown(f"**[Open in Quip]({link})**")
                    if created or updated:
                        st.caption(f"🕒 Created: {created} | Updated: {updated}")

                    preview = d.get("text", "")
                    if preview:
                        st.text_area(
                            "Preview",
                            preview[:2000] + ("..." if len(preview) > 2000 else ""),
                            height=150,
                            key=f"preview_{i}_{meta.get('id', i)}",
                        )

                    with st.expander("🔍 Metadata", expanded=False):
                        st.json(meta)

        # -------------------------
        # Download Filtered Results
        # -------------------------
        if filtered_docs:
            # Convert to DataFrame
            export_rows = []
            for d in filtered_docs:
                meta = d.get("meta", {})
                export_rows.append({
                    "Title": meta.get("title", ""),
                    "Quip Link": meta.get("link", ""),
                    "Created": meta.get("created_usec", ""),
                    "Updated": meta.get("updated_usec", ""),
                    "Preview": d.get("text", "")[:500].replace("\n", " ")  # first 500 chars
                })

            df_export = pd.DataFrame(export_rows)
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False)

            st.download_button(
                label="📥 Download filtered results (CSV)",
                data=csv_buffer.getvalue(),
                file_name="quip_corpus_filtered.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.caption("No documents to export.")  

# ---------------- Q&A with FAISS (tab2 stays as-is below) ----------------

def _ensure_index():
    """Build an in-memory FAISS index from st.session_state.docs (with chunking)."""
    import faiss, numpy as np
    if not st.session_state.get("docs"):
        st.warning("No documents in corpus yet. Ingest or crawl first.")
        return False

    embedder = _embedder()
    chunk_size = st.session_state.get("chunk_size", 1500)
    overlap    = st.session_state.get("chunk_overlap", 200)

    texts, metas = [], []
    for d in st.session_state.docs:
        txt = d.get("text","")
        if not txt:
            continue
        parts = chunk_text(txt, chunk_size=chunk_size, overlap=overlap)
        for i, p in enumerate(parts):
            texts.append(p)
            m = {**(d.get("meta") or {}), "source": d.get("source"), "chunk_id": i}
            metas.append(m)

    if not texts:
        st.error("No text available to index.")
        return False

    vecs = embedder.encode(texts).astype("float32")
    # normalize for cosine via inner product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    st.session_state.faiss_index = index
    st.session_state.faiss_texts = texts
    st.session_state.faiss_meta  = metas
    st.success(f"Indexed {len(texts)} chunks from {len(st.session_state.docs)} document(s).")
    return True


def _search(query: str, top_k: int = 5):
    """Search current FAISS index and return ranked passages with metadata."""
    import numpy as np
    if "faiss_index" not in st.session_state:
        st.warning("No index in memory. Click 'Create / Rebuild Index' first.")
        return []

    embedder = _embedder()
    qv = embedder.encode([query])[0].astype("float32")
    qv /= (np.linalg.norm(qv) + 1e-12)

    D, I = st.session_state.faiss_index.search(qv[np.newaxis, :], top_k)
    hits = []
    for rank, idx in enumerate(I[0].tolist(), 1):
        if idx == -1: 
            continue
        hits.append({
            "rank": rank,
            "text": st.session_state.faiss_texts[idx],
            "meta": st.session_state.faiss_meta[idx],
            "score": float(D[0][rank-1])
        })
    return hits


# ---------------- Q&A with FAISS ----------------
with tab2:
    st.subheader("Ask CAP Questions")

    # Show which backend is active (OpenAI or SentenceTransformers)
    try:
        be = _embedder()  # Initialize or reuse the embedder
        backend_label = f"Embedding backend: {'OpenAI (' + be.model + ')' if be.backend == 'openai' else 'Sentence-Transformers (all-MiniLM-L6-v2)'}"
        st.caption(backend_label)
    except Exception as e:
        st.warning(f"⚠️ Embedding backend not ready: {e}")
     
    colA, colB = st.columns(2)
    with colA:
        if st.button("Create / Rebuild Index"):
            _ensure_index()
            st.session_state.last_index_sig = _docs_signature()
    with colB:
        top_k = st.slider("Context passages", 3, 10, 5)

    st.markdown("---")
    # ✅ QUESTION INPUT BOX
    q = st.text_input(
        "💬 Your question",
        placeholder="e.g., What is the U-turn rule in Montenegro?",
        key="user_query"
    )
    use_llm = st.checkbox(
    "Use OpenAI to compose an answer (optional)",
    value=False,
    key="qa_use_llm"
)

if use_llm:
    openai_key = os.getenv("OPENAI_API_KEY") or st.text_input(
        "OpenAI API Key (for answer composer)",
        type="password",
        key="qa_openai_key"
    )
else:
    openai_key = None

    # ✅ Handle query
    if q:
        hits = _search(q, top_k=top_k)
        if not hits:
            st.info("No results yet. Build the index, then ask again.")
        else:
            with st.expander("Context used", expanded=False):
                for h in hits:
                    title = h["meta"].get("title") or h["meta"].get("source") or "Untitled"
                    link = h["meta"].get("link")
                    st.markdown(f"**[{h['rank']}] {title}**" + (f" — {link}" if link else ""))
                    st.write(h["text"][:800] + ("..." if len(h["text"]) > 800 else ""))

            final_answer = None
            if use_llm and openai_key:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=openai_key)
                    sources_block = "\n\n".join(
                        [f"[{i+1}] {h['meta'].get('title','Untitled')} — {h['meta'].get('link','')}\n{h['text']}"
                         for i, h in enumerate(hits)]
                    )
                    prompt = (
                        "You are CAP Assistant. Answer clearly using ONLY the numbered sources below.\n"
                        "Add citation markers like [1], [2] when you use a fact. If unsure, say you don't know.\n\n"
                        f"Question: {q}\n\nSources:\n{sources_block}\n\nAnswer (with [n] citations):"
                    )
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    final_answer = resp.choices[0].message.content
                except Exception as e:
                    st.error(f"Answer composition failed; showing top passage instead. {e}")

            if not final_answer:
                best = hits[0]
                final_answer = best["text"][:1200] + ("..." if len(best["text"]) > 1200 else "")
                final_answer += "\n\n_This is the top matching passage. Enable OpenAI to get a concise answer._"

            st.success(final_answer)

            # ✅ Sources below answer
            st.markdown("**Sources**")
            for h in hits:
                title = h["meta"].get("title") or h["meta"].get("source") or "Untitled"
                link = h["meta"].get("link")
                if link:
                    st.markdown(f"[{h['rank']}] {title} — {link}")
                else:
                    st.markdown(f"[{h['rank']}] {title}")

    # -------------------------
# Download Filtered Results (Markdown)
# -------------------------
if filtered_docs:
    # Generate a combined Markdown text for all filtered docs
    md_parts = []
    for d in filtered_docs:
        meta = d.get("meta", {})
        title = meta.get("title", "Untitled Document")
        link = meta.get("link", "")
        created = meta.get("created_usec", "")
        updated = meta.get("updated_usec", "")
        text = d.get("text", "").strip()

        md_parts.append(f"# {title}\n")
        if link:
            md_parts.append(f"[Open in Quip]({link})\n")
        if created or updated:
            md_parts.append(f"*Created: {created} | Updated: {updated}*\n\n")

        if text:
            md_parts.append(text[:5000])  # Limit to 5000 chars for file size
            if len(text) > 5000:
                md_parts.append("\n\n*(Preview truncated)*")

        md_parts.append("\n\n---\n\n")

    markdown_output = "\n".join(md_parts)

    st.download_button(
        label="📝 Download filtered results (Markdown)",
        data=markdown_output,
        file_name="quip_corpus_filtered.md",
        mime="text/markdown",
        use_container_width=True
    )
else:
    st.caption("No documents to export.")  

    # -------------------------
# Download: one Markdown file per document (ZIP)
# -------------------------
def _slugify(name: str) -> str:
    # simple, filesystem-safe slug
    name = re.sub(r"\s+", "_", name.strip())          # spaces -> underscores
    name = re.sub(r"[^\w\.-]", "", name)              # remove unsafe chars
    return name[:80] or "document"

if filtered_docs:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for d in filtered_docs:
            meta = d.get("meta", {})
            title = meta.get("title", "Untitled Document")
            link = meta.get("link", "")
            created = meta.get("created_usec", "")
            updated = meta.get("updated_usec", "")
            text = d.get("text", "").strip()

            # filename: <slug>-<id>.md if we have an id
            tid = meta.get("id", "")
            base = _slugify(title if title else (tid or "document"))
            fname = f"{base}.md" if not tid else f"{base}-{tid}.md"

            md_parts = [f"# {title}\n"]
            if link:
                md_parts.append(f"[Open in Quip]({link})\n")
            if created or updated:
                md_parts.append(f"*Created: {created} | Updated: {updated}*\n\n")
            if text:
                md_parts.append(text)
            md_content = "\n".join(md_parts)

            zf.writestr(fname, md_content)

    zip_buf.seek(0)
    st.download_button(
        label="🗂️ Download filtered docs (ZIP of Markdown files)",
        data=zip_buf.getvalue(),
        file_name="quip_corpus_filtered_md.zip",
        mime="application/zip",
        use_container_width=True,
    )
else:
    st.caption("No documents to export.")       

with st.expander("ℹ️ Tips & Notes", expanded=False):
    st.markdown("""
- For **Quip API**: use `https://platform.quip.com` or your enterprise base (e.g., `https://platform.quip-apple.com`).
- If you get `401 Unauthorized`, double‑check your token and whether your account has API access.
- The current index uses **TF‑IDF** + cosine similarity for reliability and light dependencies.
  - You can later switch to embeddings (e.g., sentence‑transformers) + FAISS/Chroma if desired.
- Rebuild the index after adding or modifying documents.
""")
