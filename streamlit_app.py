# streamlit_app.py
import os
import re
import json
import time
import random
import base64
import requests
import zipfile
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

# NEW OpenAI integration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_ENABLED = bool(OPENAI_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY) if GPT_ENABLED else None

def gpt_refine_answer(question: str, context: str) -> str | None:
    if not GPT_ENABLED:
        return None
    try:
        with st.spinner("Refining answer with GPT…"):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",  # or gpt-4o, gpt-3.5-turbo if allowed
                messages=[
                    {"role": "system", "content": "Answer using only the provided context when possible. If the context doesn't contain the full answer, make a best-effort inference based on related rules."},
                    {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}"}
                ],
                temperature=0.2,
                max_tokens=350,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"GPT unavailable ({e}); using local retrieval only.")
        return None
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

# Enhanced CSS styling for intense grey/black/white theme with thicker borders
st.markdown("""
<style>
    /* Main app background and text colors */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }

    /* Header styling with intense contrast */
    .main h1 {
        color: #000000 !important;
        font-weight: 800 !important;
        border-bottom: 4px solid #212529 !important;
        padding-bottom: 1rem !important;
        margin-bottom: 1.5rem !important;
        font-size: 2.5rem !important;
    }

    /* Subheader styling */
    .main h2, .main h3 {
        color: #212529 !important;
        font-weight: 700 !important;
        border-left: 6px solid #495057 !important;
        padding-left: 1rem !important;
        margin: 1.5rem 0 1rem 0 !important;
    }

    /* Enhanced button styling following Figma principles */
    .stButton > button {
        background-color: #212529 !important;
        color: #ffffff !important;
        border: 3px solid #212529 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    .stButton > button:hover {
        background-color: #495057 !important;
        border-color: #495057 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.25) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }

    /* Secondary button style for validation/fetch actions */
    .stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #212529 !important;
        border: 3px solid #212529 !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }

    /* Input field styling with thicker borders */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        border: 3px solid #dee2e6 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        color: #212529 !important;
        font-weight: 500 !important;
        padding: 0.75rem !important;
        transition: border-color 0.2s ease-in-out !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #212529 !important;
        box-shadow: 0 0 0 3px rgba(33, 37, 41, 0.1) !important;
        outline: none !important;
    }

    /* Sidebar styling with intense contrast */
    .css-1d391kg {
        background-color: #343a40 !important;
        color: #ffffff !important;
        border-right: 4px solid #212529 !important;
    }

    .css-1d391kg .stMarkdown {
        color: #ffffff !important;
    }

    .css-1d391kg h2, .css-1d391kg h3 {
        color: #ffffff !important;
        border-left: 4px solid #ffffff !important;
    }

    /* Sidebar input fields */
    .css-1d391kg .stTextInput > div > div > input,
    .css-1d391kg .stTextArea > div > div > textarea {
        background-color: #495057 !important;
        color: #ffffff !important;
        border: 2px solid #6c757d !important;
    }

    .css-1d391kg .stTextInput > div > div > input:focus,
    .css-1d391kg .stTextArea > div > div > textarea:focus {
        border-color: #ffffff !important;
        background-color: #495057 !important;
    }

    /* Expander styling with thick borders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        border: 3px solid #dee2e6 !important;
        border-radius: 8px !important;
        color: #212529 !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }

    .streamlit-expanderContent {
        border: 3px solid #dee2e6 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        background-color: #ffffff !important;
        padding: 1.5rem !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa !important;
        padding: 0.5rem !important;
        border-radius: 8px !important;
        border: 3px solid #dee2e6 !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff !important;
        color: #495057 !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #212529 !important;
        color: #ffffff !important;
        border-color: #212529 !important;
    }

    /* Container and card styling */
    .stContainer {
        border: 3px solid #dee2e6 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        background-color: #ffffff !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }

    /* Success/Error message styling */
    .stSuccess {
        background-color: #ffffff !important;
        border: 3px solid #198754 !important;
        color: #198754 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stError {
        background-color: #ffffff !important;
        border: 3px solid #dc3545 !important;
        color: #dc3545 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stWarning {
        background-color: #ffffff !important;
        border: 3px solid #fd7e14 !important;
        color: #fd7e14 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stInfo {
        background-color: #ffffff !important;
        border: 3px solid #0dcaf0 !important;
        color: #0dcaf0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Checkbox and slider styling */
    .stCheckbox {
        color: #212529 !important;
        font-weight: 500 !important;
    }

    .stSlider {
        color: #212529 !important;
        font-weight: 500 !important;
    }

    /* Download button specific styling */
    .stDownloadButton > button {
        background-color: #495057 !important;
        color: #ffffff !important;
        border: 3px solid #495057 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        width: 100% !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stDownloadButton > button:hover {
        background-color: #212529 !important;
        border-color: #212529 !important;
        transform: translateY(-1px) !important;
    }

    /* Caption styling for better hierarchy */
    .caption {
        color: #6c757d !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        border-left: 3px solid #dee2e6 !important;
        padding-left: 0.75rem !important;
    }

    /* Code and pre-formatted text */
    code {
        background-color: #f8f9fa !important;
        color: #212529 !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 4px !important;
        padding: 0.25rem 0.5rem !important;
        font-weight: 600 !important;
    }

    /* Metric styling */
    .metric-container {
        background-color: #ffffff !important;
        border: 3px solid #dee2e6 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        text-align: center !important;
    }

    /* JSON viewer styling */
    .stJson {
        background-color: #f8f9fa !important;
        border: 3px solid #dee2e6 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

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


def get_first_existing(d: dict, *keys):
    """
    Utility to get the first existing key from a dict, e.g. first_existing(doc, "html", "rendered_html")
    """
    for k in keys:
        if d.get(k):
            return d[k]
    return None
    # Legacy alias (backward compatibility)
def first_existing(d: dict, *keys):
    """Alias for backward compatibility."""
    return get_first_existing(d, *keys)
    
def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def http_get_with_retries(url: str, headers: dict, timeout=30, max_tries=5):
    """
    Retry on 5xx and timeouts with exponential backoff + jitter.
    Returns (ok, payload_or_error_text, status_code)
    """
    delay = 0.75
    for attempt in range(1, max_tries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code >= 500:
                # server error → retry
                pass
            elif r.status_code == 429:
                # respect rate limiting if present
                ra = r.headers.get("Retry-After")
                if ra:
                    time.sleep(int(ra))
                else:
                    time.sleep(delay)
            else:
                try:
                    return True, r.json(), r.status_code
                except Exception:
                    return True, r.text, r.status_code
        except requests.Timeout:
            # retry
            pass
        except requests.RequestException as e:
            # network issues; retry a couple of times then give up
            if attempt == max_tries:
                return False, f"RequestException: {e}", None

        if attempt < max_tries:
            # backoff + jitter
            time.sleep(delay + random.uniform(0, 0.4))
            delay *= 2

    return False, f"Max retries reached for {url}", None

def fetch_thread_metadata_safe(tid: str, base_url: str, token: str):
    """
    Fetch thread metadata with retry logic.
    """
    url = f"{base_url.rstrip('/')}/1/threads/{tid}"
    ok, data, code = http_get_with_retries(url, {"Authorization": f"Bearer {token}"})
    if not ok:
        raise RuntimeError(f"Failed metadata lookup for {tid}: {data} (status={code})")
    return data

def retry_failed_threads(failed_ids: list, quip_token: str, quip_base: str):
    """
    Helper function to retry a list of failed thread IDs.
    Returns (successful_threads, still_failed_ids)
    """
    if not failed_ids:
        return {}, []

    retry_ids = [tid for tid, _ in failed_ids]
    retry_data = fetch_quip_threads_single("\n".join(retry_ids), quip_token, quip_base)
    return retry_data.get("threads", {}), retry_data.get("failed_ids", [])

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

def fetch_all_threads_in_folder(folder_id_or_token: str, base_url: str, headers: dict, _depth=0, _seen=None) -> list:
    """
    Recursively collect all thread_ids from a folder and its subfolders.
    Tries both /1/folders/<id> and /1/folders/<id>/children (enterprise quip nuance).
    """
    import requests

    indent = "  " * _depth
    if _seen is None:
        _seen = set()

    base = base_url.rstrip("/")
    fid = folder_id_or_token.strip()

    # 1) Resolve public token -> internal id (once)
    #    If fid looks like a public token (starts with https or short token), resolve via /1/folders/<token>
    if fid not in _seen:
        meta_url = f"{base}/1/folders/{fid}"
        r = requests.get(meta_url, headers=headers, timeout=30)
        if r.status_code == 200:
            meta = r.json().get("folder", {})
            internal_id = meta.get("id") or fid
            if internal_id != fid:
                st.write(f"{indent}Resolved → {internal_id}")
            fid = internal_id
        else:
            st.warning(f"{indent}⚠️ Folder resolve failed for {fid}: {r.status_code} {r.text[:150]}")

    # Prevent cycles
    if fid in _seen:
        return []
    _seen.add(fid)

    all_threads = []

    # 2) Try to read children directly from /1/folders/<id>
    meta_url2 = f"{base}/1/folders/{fid}"
    r2 = requests.get(meta_url2, headers=headers, timeout=30)
    children = []
    if r2.status_code == 200:
        folder_data = r2.json().get("folder", {})
        children = folder_data.get("children", []) or []
    else:
        st.warning(f"{indent}⚠️ Folder {fid}: {r2.status_code} {r2.text[:150]}")

    # 3) If empty, try /1/folders/<id>/children (common on enterprise)
    if not children:
        child_url = f"{base}/1/folders/{fid}/children"
        r3 = requests.get(child_url, headers=headers, timeout=30)
        if r3.status_code == 200:
            j = r3.json()
            # Some tenants return {"children":[...]} or just the array
            children = j.get("children", j if isinstance(j, list) else [])
        else:
            st.warning(f"{indent}⚠️ Children endpoint for {fid}: {r3.status_code} {r3.text[:150]}")

    # 4) Parse children
    threads_here = 0
    subs_here = 0
    for child in children:
        ctype = child.get("type")
        if ctype == "thread" and child.get("thread_id"):
            all_threads.append(child["thread_id"])
            threads_here += 1
        elif ctype == "folder" and child.get("folder_id"):
            subs_here += 1
            sub_id = child["folder_id"]
            all_threads.extend(
                fetch_all_threads_in_folder(sub_id, base_url, headers, _depth=_depth+1, _seen=_seen)
            )
        # some tenants use 'id' for folder child id
        elif ctype == "folder" and child.get("id"):
            subs_here += 1
            sub_id = child["id"]
            all_threads.extend(
                fetch_all_threads_in_folder(sub_id, base_url, headers, _depth=_depth+1, _seen=_seen)
            )

    st.write(f"{indent}Found {threads_here} thread(s) and {subs_here} subfolder(s) in {fid}.")
    return all_threads

def fetch_quip_threads_single(thread_ids_or_tokens: str, token: str, base_url: str) -> dict:
    """
    Resolves each supplied token (public link or internal id) and fetches
    full thread data directly via /1/threads/<id> with retry logic and fail-soft behavior.
    Returns a dict shaped like {'threads': {id: thread_obj}, 'failed_ids': [(id, error_msg)]}
    """
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token.strip()}"}

    # --- resolve public tokens to internal ids ---
    ids = [i.strip() for i in thread_ids_or_tokens.strip().splitlines() if i.strip()]
    resolved = {}
    failed_ids = []

    for token_or_id in ids:
        try:
            # Metadata lookup with retry
            meta_url = f"{base}/1/threads/{token_or_id}"
            ok, data, status = http_get_with_retries(meta_url, headers)
            if not ok:
                failed_ids.append((token_or_id, f"Metadata lookup failed: {data}"))
                continue

            if isinstance(data, str):
                # Try to parse as JSON if it's a string
                try:
                    data = json.loads(data)
                except:
                    failed_ids.append((token_or_id, f"Invalid JSON response: {data[:200]}"))
                    continue

            meta = data.get("thread")
            if not meta:
                failed_ids.append((token_or_id, "No thread metadata in response"))
                continue

            thread_id = meta.get("id", token_or_id)

            # --- fetch full thread (same endpoint) with retry ---
            full_url = f"{base}/1/threads/{thread_id}?include_html=true"
            ok2, data2, status2 = http_get_with_retries(full_url, headers)
            if not ok2:
                failed_ids.append((thread_id, f"Full thread fetch failed: {data2}"))
                continue

            if isinstance(data2, str):
                try:
                    data2 = json.loads(data2)
                except:
                    failed_ids.append((thread_id, f"Invalid JSON in full thread response: {data2[:200]}"))
                    continue

            doc = data2.get("thread", {})
            if doc:
                resolved[thread_id] = doc
            else:
                failed_ids.append((thread_id, "No thread data in full response"))

        except Exception as e:
            failed_ids.append((token_or_id, f"Exception: {str(e)}"))

    return {"threads": resolved, "failed_ids": failed_ids}


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
# ---------- Persistence: corpus + vectors + FAISS ----------
PERSIST_DIR = "cap_artifacts"
DOCS_JSONL = os.path.join(PERSIST_DIR, "docs.jsonl")
VECTORS_NPY = os.path.join(PERSIST_DIR, "embeddings.npy")
METAS_PKL  = os.path.join(PERSIST_DIR, "emb_meta.pkl")
FAISS_IDX  = os.path.join(PERSIST_DIR, "faiss.index")

def _ensure_dir():
    os.makedirs(PERSIST_DIR, exist_ok=True)

def save_corpus_and_index():
    _ensure_dir()
    with open(DOCS_JSONL, "w", encoding="utf-8") as f:
        for d in st.session_state.get("docs", []):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    if "faiss_index" in st.session_state and st.session_state.get("faiss_texts") is not None:
        import faiss
        faiss.write_index(st.session_state.faiss_index, FAISS_IDX)
        np.save(VECTORS_NPY, np.array([], dtype=np.float32))
        with open(METAS_PKL, "wb") as f:
            pickle.dump({
                "texts": st.session_state.faiss_texts,
                "metas": st.session_state.faiss_meta,
                "dim": st.session_state.faiss_index.d,
            }, f)
        return True
    return True

def load_corpus_and_index():
    ok_docs = False
    ok_index = False
    docs = []

    if os.path.exists(DOCS_JSONL):
        with open(DOCS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    docs.append(json.loads(line))
                except Exception:
                    pass
    if docs:
        st.session_state.docs = docs
        ok_docs = True

    if os.path.exists(FAISS_IDX) and os.path.exists(METAS_PKL):
        import faiss
        try:
            index = faiss.read_index(FAISS_IDX)
            with open(METAS_PKL, "rb") as f:
                payload = pickle.load(f)
            st.session_state.faiss_index = index
            st.session_state.faiss_texts = payload.get("texts", [])
            st.session_state.faiss_meta  = payload.get("metas", [])
            ok_index = True
        except Exception as e:
            st.warning(f"Failed to load FAISS index: {e}")
    return ok_docs, ok_index

def clear_corpus_and_index():
    st.session_state.pop("docs", None)
    st.session_state.pop("faiss_index", None)
    st.session_state.pop("faiss_texts", None)
    st.session_state.pop("faiss_meta", None)
    for p in [DOCS_JSONL, VECTORS_NPY, METAS_PKL, FAISS_IDX]:
        try:
            if os.path.exists(p): os.remove(p)
        except Exception:
            pass

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


def _search(query: str, top_k: int = 10):
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
def _extract_token(maybe_url_or_token: str) -> str:
    s = maybe_url_or_token.strip()
    if s.startswith("http"):
        m = re.search(r"/([A-Za-z0-9]{8,})$", s.split("?")[0].rstrip("/"))
        return m.group(1) if m else s
    return s

def resolve_folder_internal_id(maybe_url_or_token: str, base_url: str, headers: dict) -> str | None:
    """
    Accepts a Quip folder share URL or token and returns the INTERNAL folder id.
    e.g. 'https://quip-apple.com/Rc0dOerFFlFP' or 'Rc0dOerFFlFP' -> 'SLa9OAx6077'
    """
    token = _extract_token(maybe_url_or_token)
    r = requests.get(f"{base_url.rstrip('/')}/1/folders/{token}", headers=headers, timeout=30)
    if r.status_code == 200 and r.json().get("folder", {}).get("id"):
        return r.json()["folder"]["id"]
    return None

def list_children_for_folder(internal_id: str, base_url: str, headers: dict) -> list[dict]:
    """
    Returns a list of child descriptors. Tries both inline and /children endpoint.
    """
    base = base_url.rstrip("/")
    # 1) Try inline 'children'
    r = requests.get(f"{base}/1/folders/{internal_id}", headers=headers, timeout=30)
    if r.status_code == 200:
        folder = r.json().get("folder", {})
        ch = folder.get("children")
        if isinstance(ch, list) and ch:
            return ch

    # 2) Fallback: explicit children endpoint
    r2 = requests.get(f"{base}/1/folders/{internal_id}/children", headers=headers, timeout=30)
    if r2.status_code == 200:
        data = r2.json()
        # Some tenants return {'children': [...]} others return a bare list
        if isinstance(data, dict) and isinstance(data.get("children"), list):
            return data["children"]
        if isinstance(data, list):
            return data

    # If both fail, return empty
    return []

def walk_folder_threads_recursive(internal_id: str, base_url: str, headers: dict) -> list[str]:
    """
    Recursively collect all thread IDs from a folder tree.
    """
    out = []
    seen_folders = set()

    def _walk(fid: str):
        if fid in seen_folders:
            return
        seen_folders.add(fid)

        children = list_children_for_folder(fid, base_url, headers)
        for c in children:
            ctype = c.get("type")
            if ctype == "thread" and c.get("thread_id"):
                out.append(c["thread_id"])
            elif ctype == "folder" and c.get("folder_id"):
                _walk(c["folder_id"])
            # Some tenants use 'subfolder' or other labels; be liberal:
            elif c.get("folder_id"):
                _walk(c["folder_id"])

    _walk(internal_id)
    # de-dupe
    uniq = []
    seen = set()
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

    return hits
    if st.button("Fetch Folder Contents", key="fetch_folders_btn"):
       if not quip_token or not folder_ids.strip():
        st.error("Provide API token and at least one folder ID/token.")
    else:
        headers = {"Authorization": f"Bearer {quip_token.strip()}"}
        base = quip_base.rstrip("/")

        all_threads = []
        for raw in [x.strip() for x in folder_ids.splitlines() if x.strip()]:
            # Accept share token (Rc0dOerFFlFP) or internal id (e.g. SLa9OAx6077)
            fid = resolve_folder_id(raw, base, headers)
            if not fid:
                st.warning(f"Could not resolve folder token/id: {raw}")
                continue

            tids = walk_folder_threads_recursive(fid, base, headers)
            st.write(f"📁 Folder {raw} → internal {fid}: found {len(tids)} thread(s)")
            all_threads.extend(tids)

        # De-dupe
        all_threads = list(dict.fromkeys(all_threads))
        if not all_threads:
            st.warning("Found 0 unique thread(s) across folder(s).")
            # Optional: show debug of the first folder JSON
            # with st.expander("Debug: first folder JSON", expanded=False):
            #     st.json(requests.get(f"{base}/1/folders/{fid}", headers=headers, timeout=30).json())
        else:
            st.success(f"Found {len(all_threads)} unique thread(s) across folder(s). Fetching…")
            data = fetch_quip_threads_single("\n".join(all_threads), quip_token, quip_base)

            threads = data.get("threads") or {}
            failed_ids = data.get("failed_ids", [])
            num_added = 0
            for tid, doc in threads.items():
                html = (doc.get("html") or doc.get("rendered_html") or
                        doc.get("document_html") or doc.get("content_html") or
                        doc.get("body_html") or doc.get("full_html") or "")
                text = html_to_text(html) if html else (doc.get("title") or "")

                meta = {
                    "id": doc.get("id") or tid,
                    "title": doc.get("title"),
                    "link": doc.get("link"),
                    "updated_usec": doc.get("updated_usec"),
                    "created_usec": doc.get("created_usec"),
                }
                st.session_state.docs.append({"source": f"quip:{tid}", "text": text, "meta": meta})
                num_added += 1

                st.success(f"Added {num_added} threads to the corpus.")

                # Show failed threads if any
                if failed_ids:
                    st.warning(f"{len(failed_ids)} thread(s) failed (e.g., 504, timeout). Check the expandable list below.")
                    with st.expander("Show failed thread IDs", expanded=False):
                        for tid, err in failed_ids:
                            st.code(f"{tid}  →  {err}")
# -------------------------------
# Sidebar: Ingest & Quip Controls
# -------------------------------
def render_sidebar():
    with st.sidebar:
        st.header("Ingest Documents")
        st.caption("Add content from Quip or upload docs. Persistence lets you save/load corpus + FAISS index.")
        st.markdown("🟢 _Sidebar loaded_")

        # ✅ Single toggle (unique key)
        st.checkbox(
            "⚡ Auto-rebuild index when corpus changes",
            value=True,
            key="sidebar_auto_index",   # <— unique key, do NOT reuse elsewhere
            help="If on, FAISS is rebuilt automatically whenever documents are added/updated."
        )

        # 💾 Persistence
        with st.expander("💾 Persistence (save / load)", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Save", key="persist_save_btn"):
                    try:
                        ok = save_corpus_and_index()
                        st.success("Saved corpus + index to disk." if ok else "Nothing to save.")
                    except Exception as e:
                        st.error(f"Save failed: {e}")
            with c2:
                if st.button("Load", key="persist_load_btn"):
                    try:
                        ok_docs, ok_idx = load_corpus_and_index()
                        st.success(f"Loaded docs: {ok_docs}, index: {ok_idx}")
                    except Exception as e:
                        st.error(f"Load failed: {e}")
            with c3:
                if st.button("Clear", key="persist_clear_btn"):
                    try:
                        clear_corpus_and_index()
                        st.warning("Cleared corpus + index.")
                    except Exception as e:
                        st.error(f"Clear failed: {e}")

        # 🔗 Fetch from Quip API
        with st.expander("🔗 Fetch from Quip API", expanded=False):
            quip_base = st.text_input(
                "Quip Platform Base URL",
                value=st.session_state.get("quip_base", "https://platform.quip-apple.com"),
                help="Enterprise instances may use https://platform.quip-apple.com or similar.",
                key="quip_base_input",
            )
            quip_token = st.text_input(
                "Quip API Token",
                type="password",
                value=st.session_state.get("quip_token", ""),
                key="quip_token_input",
            )
            thread_ids = st.text_area(
                "Thread IDs (one per line)",
                placeholder="yEZuAgiIXp8V\ndCQ9AAjmKUk\n...",
                key="thread_ids_input",
            )

            # Persist latest values into session_state (use once here)
            st.session_state["quip_base"] = quip_base.strip()
            st.session_state["quip_token"] = quip_token.strip()

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
                    if not quip_token.strip() or not thread_ids.strip():
                        st.error("Provide API token and at least one thread ID.")
                    else:
                        ids = [i.strip() for i in thread_ids.strip().splitlines() if i.strip()]
                        try:
                            data = fetch_quip_threads_single("\n".join(ids), quip_token, quip_base)
                            threads = data.get("threads") or {}
                            failed_ids = data.get("failed_ids", [])

                            if not threads and "thread" in data:
                                t = data["thread"]; threads = {t.get("id", "unknown"): t}

                            if not threads and not failed_ids:
                                st.warning("No threads returned. Check IDs and access.")
                            elif threads:
                                def get_first_existing(d: dict, *keys):
                                    for k in keys:
                                        if d.get(k): return d[k]
                                    return None

                                added = 0
                                for tid, doc in threads.items():
                                    html = first_existing(
                                        doc, "html","rendered_html","document_html","content_html","body_html","full_html"
                                    )
                                    text = html_to_text(html) if html else (doc.get("title","") or "")
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
                                        "meta": meta,
                                    })
                                    added += 1
                                st.success(f"Added {added} thread(s) to the corpus.")

                                # Show failed threads if any
                                if failed_ids:
                                    st.warning(f"{len(failed_ids)} thread(s) failed (e.g., 504, timeout). You can retry just the failed ones.")
                                    with st.expander("Show failed thread IDs", expanded=False):
                                        for tid, err in failed_ids:
                                            st.code(f"{tid}  →  {err}")

                                    # Optional: retry button for failed IDs
                                    if st.button(f"Retry {len(failed_ids)} failed threads", key="retry_failed_threads"):
                                        retry_ids = [tid for tid, _ in failed_ids]
                                        retry_data = fetch_quip_threads_single("\n".join(retry_ids), quip_token, quip_base)
                                        retry_threads = retry_data.get("threads", {})
                                        retry_failed = retry_data.get("failed_ids", [])

                                        retry_added = 0
                                        for tid, doc in retry_threads.items():
                                            html = first_existing(
                                                doc, "html","rendered_html","document_html","content_html","body_html","full_html"
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
                                                "meta": meta,
                                            })
                                            retry_added += 1

                                        if retry_added > 0:
                                            st.success(f"Retry: Added {retry_added} more thread(s)!")
                                        if retry_failed:
                                            st.error(f"Still failed: {len(retry_failed)} thread(s)")
                        except Exception as e:
                            st.error(f"Error fetching threads: {e}")

        # 🕷️ Crawl linked Quip docs
        with st.expander("🕷️ Crawl linked Quip docs", expanded=False):
            seed_input = st.text_area(
                "Seed Quip doc URLs or tokens (one per line)",
                key="crawl_seed",
                placeholder="https://quip-apple.com/yEZuAgiIXp8V\n...",
                help="Paste CAP index/reference document URLs or tokens — the crawler follows Quip links and ingests linked docs."
            )
            max_pages = st.number_input(
                "Max pages to crawl",
                min_value=1, max_value=500, value=50, step=10,
                key="crawl_max"
            )

            if st.button("Crawl & Ingest", key="crawl_btn"):
                quip_token = st.session_state.get("quip_token", "")
                quip_base  = st.session_state.get("quip_base", "https://platform.quip-apple.com")
                if not quip_token or not seed_input.strip():
                    st.error("Provide API token and at least one seed URL/token.")
                else:
                    headers = {"Authorization": f"Bearer {quip_token.strip()}"}
                    seeds = [s for s in seed_input.splitlines() if s.strip()]
                    try:
                        found = crawl_quip_links(seeds, quip_base, headers, max_pages=int(max_pages))

                        def get_first_existing(d: dict, *keys):
                            for k in keys:
                                if d.get(k): return d[k]
                            return None

                        num_added = 0
                        for tid, doc in found.items():
                            html = first_existing(
                                doc, "html","rendered_html","document_html","content_html","body_html","full_html"
                            )
                            text = html_to_text(html) if html else (doc.get("title","") or "")
                            meta = {
                                "id": doc.get("id") or tid,
                                "title": doc.get("title"),
                                "link": doc.get("link"),
                                "updated_usec": doc.get("updated_usec"),
                                "created_usec": doc.get("created_usec"),
                                "_has_html": bool(html),
                            }
                            st.session_state.docs.append({"source": f"quip:{tid}", "text": text, "meta": meta})
                            num_added += 1
                        st.success(f"Crawled {len(found)} doc(s). Added {num_added} to the corpus.")
                        with st.expander("🔍 Debug: Crawl Results (IDs & Titles)", expanded=False):
                            st.json({k: found[k].get("title") for k in found})
                    except Exception as e:
                        st.error(f"Crawl error: {e}")

        # 📂 Export entire Quip folder (recursive)
        debug_folder_api = st.checkbox("🔎 Show folder API debug", value=False, key="folder_api_debug")
        with st.expander("📂 Export entire Quip folder", expanded=False):
            quip_base  = st.session_state.get("quip_base") or "https://platform.quip-apple.com"
            quip_token = st.session_state.get("quip_token") or ""

            folder_inputs = st.text_area(
                "Folder URLs or tokens (one per line)",
                key="folder_inputs",
                placeholder="https://quip-apple.com/Rc0dOerFFlFP\npqLyOdNS6PZN\n...",
                help="Paste folder URLs or public tokens. We resolve to internal folder id and recurse into subfolders.",
            )
            max_nodes = st.number_input(
                "Max items to traverse", min_value=50, max_value=2000, value=500, step=50, key="folder_max_nodes"
            )

            # Helpers
            def _extract_token_from_url(s: str) -> str:
                s = s.strip()
                if not s: return ""
                if s.startswith("http"):
                    m = re.search(r"https?://[^/]+/([A-Za-z0-9]{8,})", s)
                    return m.group(1) if m else s
                return s

            def _resolve_folder_internal_id(token_or_url: str, base_url: str, headers: dict) -> tuple[str | None, str | None]:
                tok = _extract_token_from_url(token_or_url)
                if not tok: return None, None
                r = requests.get(f"{base_url.rstrip('/')}/1/folders/{tok}", headers=headers, timeout=30)
                if r.status_code != 200:
                    st.warning(f"Resolve failed for '{tok}': {r.status_code} {r.text[:120]}")
                    return None, None
                folder = (r.json() or {}).get("folder", {})
                return folder.get("id"), tok

            def _fetch_children_for_folder(internal_id: str, public_token: str | None, base_url: str, headers: dict) -> list[dict]:
                base = base_url.rstrip('/')
                attempts = [
                    (public_token and f"{base}/1/folders/{public_token}?include_children=true", "token?include_children"),
                    (f"{base}/1/folders/{internal_id}?include_children=true", "id?include_children"),
                    (f"{base}/1/folders/{internal_id}?expanded=true", "id?expanded"),
                    (f"{base}/1/folders/{internal_id}", "id"),
                    (f"{base}/1/folders/{internal_id}/children", "id/children"),
                ]
                for url, label in [(u, lbl) for u, lbl in attempts if u]:
                    r = requests.get(url, headers=headers, timeout=30)
                    if debug_folder_api:
                       st.write(f"GET {label}: {url} → {r.status_code}")
                    if r.status_code != 200:
                         continue
                    try:
                       j = r.json()
                    except Exception:
                       continue
               
                    # 1) direct list shape: [{"type":"thread","thread_id":...}, ...]
                    if isinstance(j, list) and j and isinstance(j[0], dict):
                       if any(("thread_id" in x) or ("folder_id" in x) for x in j):
                        if debug_folder_api:
                            st.code(json.dumps(j[:3], indent=2))
                        return j

                    # 2) common envelope: {"folder": {"children": [...]}}
                    if isinstance(j, dict):
                       folder = j.get("folder") 
                       if isinstance(folder, dict) and isinstance(folder.get("children"), list):
                        children = folder["children"]
                        if debug_folder_api:
                           st.code(json.dumps(children[:3], indent=2))
                        return children

                     # 3) other possible shapes (seen on some enterprise tenants)
                    for key_name in ("children", "folder_children", "items", "nodes"):
                        val = j.get(key_name)
                        if isinstance(val, list):
                                if debug_folder_api:
                                   st.code(json.dumps(val[:3], indent=2))
                                return val

                    if debug_folder_api:
                        st.write("No children found in this response shape:")
                        st.code(json.dumps(j, indent=2))

                return []

            def _collect_all_thread_ids(root_internal_id: str, public_token: str | None, base_url: str, headers: dict, limit: int) -> list[str]:
                queue = [(root_internal_id, public_token)]
                seen_folders: set[str] = set()
                thread_ids: list[str] = []
                while queue and (len(seen_folders) + len(thread_ids)) < limit:
                    fid, tok = queue.pop(0)
                    if fid in seen_folders:
                        continue
                    seen_folders.add(fid)
                    children = _fetch_children_for_folder(fid, tok, base_url, headers)
                    for ch in children:
                        if not isinstance(ch, dict):
                           continue
                        if "thread_id" in ch and ch["thread_id"]:
                            thread_ids.append(ch["thread_id"])
                        elif "folder_id" in ch and ch["folder_id"]:
                             queue.append((ch["folder_id"], None))

              # de-dupe
                return list(dict.fromkeys(thread_ids))
                # Optional toggle: rebuild index once after import
                            # --- action ---
            reindex_after = st.checkbox(
                "Rebuild index after import (one shot)", 
                value=True, 
                key="folder_reindex_once"
            )
            fetch_clicked = st.button("Fetch Folder Contents", key="fetch_folder_btn")
            if fetch_clicked:
               if not quip_token or not folder_inputs.strip():
                  st.error("Provide API token and at least one folder URL/token.")
               else:
                    headers = {"Authorization": f"Bearer {quip_token.strip()}"}
                    base = quip_base.rstrip("/")

                    # Parse lines
                    lines = [ln.strip() for ln in folder_inputs.splitlines() if ln.strip()]

                    # Collect all thread IDs from folders
                    total_threads: list[str] = []
                    for line in lines:
                        internal_id, pub_tok = _resolve_folder_internal_id(line, base, headers)
                        if not internal_id:
                            continue
                        st.write(f"Resolved → **{internal_id}**")
                        tids = _collect_all_thread_ids(internal_id, pub_tok, base, headers, int(max_nodes))
                        st.write(f"Found **{len(tids)}** thread(s) in this folder.")
                        total_threads.extend(tids)

                    total_threads = list(dict.fromkeys(total_threads))  # Remove duplicates
                    st.success(f"Found **{len(total_threads)}** unique thread(s) across folder(s).")

                    if not total_threads:
                        st.stop()

                    # Pause auto-index during bulk import
                    prev_pause = st.session_state.get("auto_index_paused", False)
                    st.session_state["auto_index_paused"] = True

                    added = 0
                    all_failed_ids = []  # Track failed IDs across all batches
                    pbar = st.progress(0, text="Fetching threads in batches…")
                    batch_size = 20

                    try:
                        for start in range(0, len(total_threads), batch_size):
                            batch = total_threads[start:start + batch_size]

                            # Fetch a batch of threads
                            data = fetch_quip_threads_single("\n".join(batch), quip_token, quip_base)
                            threads = data.get("threads") or {}
                            batch_failed = data.get("failed_ids", [])

                            # Collect failed IDs across all batches
                            all_failed_ids.extend(batch_failed)

                            if not threads and "thread" in data:
                                t = data["thread"]
                                threads = {t.get("id", "unknown"): t}

                            # Ingest into session_state.docs
                            for tid, doc in threads.items():
                                html = first_existing(
                                    doc, "html", "rendered_html", "document_html",
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
                                st.session_state.docs.append({"source": f"quip:{tid}", "text": text, "meta": meta})
                                added += 1

                            # Update progress bar
                            done = min(start + len(batch), len(total_threads))
                            pbar.progress(
                                min(1.0, done / len(total_threads)),
                                text=f"Ingested {done}/{len(total_threads)}…"
                            )

                        pbar.empty()
                        st.success(f"Added **{added}** document(s) to the corpus.")

                        # Show failed threads if any
                        if all_failed_ids:
                            st.warning(f"{len(all_failed_ids)} thread(s) failed during batch processing (e.g., 504, timeout).")
                            with st.expander("Show failed thread IDs", expanded=False):
                                for tid, err in all_failed_ids:
                                    st.code(f"{tid}  →  {err}")

                    except Exception as e:
                        pbar.empty()
                        st.error(f"Fetch threads failed: {e}")

                    finally:
                        # Unpause auto-index flag
                        st.session_state["auto_index_paused"] = prev_pause

                    # Optional one-shot rebuild
                    if reindex_after:
                        try:
                            if _ensure_index():
                                if "last_index_sig" in st.session_state:
                                    st.session_state.last_index_sig = _docs_signature()
                                st.toast("Index rebuilt after import ✅", icon="✅")
                        except Exception as e:
                            st.warning(f"Index rebuild failed: {e}")

        # ⬆️ Upload local files
        with st.expander("⬆️ Upload local files (TXT / HTML / PDF)", expanded=False):
            up = st.file_uploader(
                "Choose files", type=["txt","html","htm","pdf"], accept_multiple_files=True, key="uploader"
            )
            if up:
                added = 0
                for f in up:
                    try:
                        if f.type in ("text/plain",) or f.name.lower().endswith(".txt"):
                            text = f.read().decode("utf-8", errors="ignore")
                        elif f.name.lower().endswith((".html", ".htm")):
                            html = f.read().decode("utf-8", errors="ignore")
                            text = html_to_text(html)
                        elif f.name.lower().endswith(".pdf"):
                            text = pdf_to_text(f) if HAS_PYPDF else ""
                        else:
                            text = ""

                        text = clean_text(text)
                        if not text:
                            st.warning(f"Skipped empty or unsupported file: {f.name}")
                            continue

                        meta = {"title": f.name, "link": "", "created_usec": None, "updated_usec": None}
                        st.session_state.docs.append({"source": f"upload:{f.name}", "text": text, "meta": meta})
                        added += 1
                    except Exception as e:
                        st.warning(f"Failed to ingest {f.name}: {e}")

                if added:
                    st.success(f"Added {added} uploaded file(s) to the corpus.")

        st.caption("🟢 Sidebar ready")

# ───────────────────────────────────────────────
# Auto-index maintenance (outside the sidebar)
# ───────────────────────────────────────────────
if "last_index_sig" not in st.session_state:
    st.session_state.last_index_sig = None

try:
    current_sig = _docs_signature()
    # Note: uses the unique sidebar key
    if (st.session_state.get("sidebarauto_index", False)
        and not st.session_state.get("auto_index_paused", False)  # NEW: not paused
        and current_sig
    ):
        if st.session_state.last_index_sig != current_sig:
            if _ensure_index():
                st.session_state.last_index_sig = current_sig
                st.toast("Index rebuilt for latest corpus ✅", icon="✅")
except Exception as e:
    st.warning(f"Auto-index check failed: {e}")

# (Optional) 🧪 Mock data expander (outside the sidebar)
with st.expander("🧪 Mock data (for testing)", expanded=False):
    if st.button("Add Montenegro mock doc"):
        st.session_state.setdefault("docs", [])
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
                "updated_usec": "20250102000000",
            }
        })
        st.success("✅ Mock doc added. Go to ❓ Ask Questions and click 'Create / Rebuild Index'.")
# after all helper functions and before tabs
try:
    render_sidebar()
except Exception as e:
    import traceback
    st.sidebar.error(f"Sidebar crash: {e}")
    st.sidebar.code(traceback.format_exc())
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

                    with st.popover("🔍 Metadata"):
                        st.json(meta, expanded=False)

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


def _search(query: str, top_k: int = 10):
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

            # retrieved_text = concat of top-k passages from FAISS (already done in your code)
            question = q
            retrieved_text = "\n\n".join([h['text'] for h in hits])

            answer = gpt_refine_answer(question, retrieved_text) or retrieved_text[:1200]

            # Fallback for manual API key entry (backward compatibility)
            if not answer and use_llm and openai_key:
                try:
                    from openai import OpenAI
                    client_local = OpenAI(api_key=openai_key)
                    sources_block = "\n\n".join(
                        [f"[{i+1}] {h['meta'].get('title','Untitled')} — {h['meta'].get('link','')}\n{h['text']}"
                         for i, h in enumerate(hits)]
                    )
                    prompt = (
                        "You are CAP Assistant. Answer clearly using ONLY the numbered sources below.\n"
                        "Add citation markers like [1], [2] when you use a fact. If unsure, say you don't know.\n\n"
                        f"Question: {q}\n\nSources:\n{sources_block}\n\nAnswer (with [n] citations):"
                    )
                    resp = client_local.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    answer = resp.choices[0].message.content
                except Exception as e:
                    st.error(f"Answer composition failed; showing top passage instead. {e}")

            final_answer = answer or retrieved_text[:1000]

            st.markdown("### 🧭 Answer")
            st.write(final_answer)

            # Optional: show a subtle banner so users know the mode
            st.caption("GPT refinement: ON" if GPT_ENABLED else "GPT refinement: OFF (local retrieval only)")

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
