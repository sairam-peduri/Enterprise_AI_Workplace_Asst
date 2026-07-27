"""Lazy, persistent PDF retrieval for the knowledge agent."""

from functools import lru_cache
from pathlib import Path

import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"


class LocalOllamaEmbeddings(Embeddings):
    """Small-batch Ollama embedding adapter compatible with Chroma.

    The installed Ollama server can reject the very large batch sent by the
    LangChain adapter while indexing all policy PDFs. Batching locally keeps
    indexing reliable without changing the local inference backend.
    """

    model: str = "nomic-embed-text"
    batch_size: int = 8

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        response = requests.post(
            "http://127.0.0.1:11434/api/embed",
            json={"model": self.model, "input": inputs, "keep_alive": "10m"},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


@lru_cache(maxsize=1)
def _get_or_create_vector_store():
    """Build the Chroma collection on first use, rather than during import."""
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError(
            "RAG dependencies are missing. Run `pip install -r requirements.txt`."
        ) from exc

    embeddings = LocalOllamaEmbeddings()
    if CHROMA_PERSIST_DIR.exists() and any(CHROMA_PERSIST_DIR.iterdir()):
        vector_store = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR), embedding_function=embeddings
        )
        # An interrupted first indexing run creates an empty SQLite store. Do not
        # mistake it for a ready knowledge base.
        if vector_store._collection.count() > 0:
            return vector_store
        vector_store.delete_collection()

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No policy PDFs were found in the data directory.")

    documents = []
    for pdf_path in pdf_files:
        documents.extend(PyPDFLoader(str(pdf_path)).load())
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1_000, chunk_overlap=200
    ).split_documents(documents)
    try:
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )
    except Exception:
        # Do not retain a partial collection after an interrupted embedding run.
        if CHROMA_PERSIST_DIR.exists():
            incomplete = Chroma(
                persist_directory=str(CHROMA_PERSIST_DIR), embedding_function=embeddings
            )
            incomplete.delete_collection()
        raise


@tool
def search_policy(query: str) -> str:
    """Search the official company policy PDFs and return cited source excerpts."""
    try:
        results = _get_or_create_vector_store().similarity_search(query, k=3)
    except Exception as exc:
        return f"Knowledge base unavailable: {exc}"
    if not results:
        return "No relevant information found in the company documents."

    excerpts = []
    for document in results:
        source = Path(document.metadata.get("source", "Unknown")).name
        page = int(document.metadata.get("page", 0)) + 1
        excerpts.append(f"--- Source: {source} (Page {page}) ---\n{document.page_content}")
    return "\n\n".join(excerpts)
