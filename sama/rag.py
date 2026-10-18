"""Small RAG layer: ingest local files and GitHub repos into Chroma, then retrieve."""
import io
import os
import zipfile
from typing import Dict, List

import chromadb
import requests
from sentence_transformers import SentenceTransformer

PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY", "db")
EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_FILE_BYTES = 1_000_000

TEXT_EXTS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp",
    ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".yml", ".yaml", ".json", ".toml",
    ".ini", ".cfg", ".html", ".css", ".sql", ".csv", ".log",
}

_embedder = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDINGS_MODEL_NAME)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        _collection = client.get_or_create_collection("sama_docs")
    return _collection


def read_bytes(name: str, data: bytes) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext == ".pdf":
        import fitz  # PyMuPDF
        with fitz.open(stream=data, filetype="pdf") as pdf:
            return "\n".join(page.get_text() for page in pdf)
    if ext == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    if ext in TEXT_EXTS:
        return data.decode("utf-8", errors="ignore")
    return ""


def chunk(text: str) -> List[str]:
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step) if text[i:i + CHUNK_SIZE].strip()]


def add_text(source: str, text: str) -> int:
    chunks = chunk(text)
    if not chunks:
        return 0
    col = _get_collection()
    embeddings = _get_embedder().encode(chunks).tolist()
    ids = [f"{source}::{i}" for i in range(len(chunks))]
    col.upsert(ids=ids, documents=chunks, embeddings=embeddings,
               metadatas=[{"source": source}] * len(chunks))
    return len(chunks)


def add_file(name: str, data: bytes) -> int:
    return add_text(name, read_bytes(name, data))


def add_local_folder(folder: str) -> int:
    total = 0
    for root, _, files in os.walk(folder):
        if any(part.startswith(".") for part in root.split(os.sep) if part not in (".", "")):
            continue
        for f in files:
            path = os.path.join(root, f)
            if os.path.getsize(path) > MAX_FILE_BYTES:
                continue
            with open(path, "rb") as fh:
                total += add_file(os.path.relpath(path, folder), fh.read())
    return total


def add_github_repo(repo: str, ref: str = "", token: str = "") -> int:
    """repo is 'owner/name' or a github.com URL. Token is only needed for private repos."""
    repo = repo.replace("https://github.com/", "").strip("/").removesuffix(".git")
    headers = {"Accept": "application/vnd.github+json"}
    token = token or os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/repos/{repo}/zipball/{ref}".rstrip("/")
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    total = 0
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for info in zf.infolist():
            if info.is_dir() or info.file_size > MAX_FILE_BYTES:
                continue
            rel = info.filename.split("/", 1)[-1]
            total += add_text(f"{repo}/{rel}", read_bytes(rel, zf.read(info)))
    return total


def retrieve(query: str, k: int = 4) -> List[Dict[str, str]]:
    col = _get_collection()
    if col.count() == 0:
        return []
    emb = _get_embedder().encode([query]).tolist()
    res = col.query(query_embeddings=emb, n_results=k)
    return [{"source": m["source"], "content": d}
            for d, m in zip(res["documents"][0], res["metadatas"][0])]
