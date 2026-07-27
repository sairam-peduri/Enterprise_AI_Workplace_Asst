import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

# Dynamically locate the data/ directory and setup a path for Chroma to save its DB
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

def _get_or_create_vector_store():
    """
    Loads existing Chroma DB or processes the 4 PDFs to create a new one.
    """
    # Initialize the local Ollama embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # If the database already exists, just load it
    if CHROMA_PERSIST_DIR.exists() and any(CHROMA_PERSIST_DIR.iterdir()):
        return Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings
        )
    
    # Otherwise, read the PDFs from the data/ folder
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}")
        
    docs = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
        
    # Split the documents into manageable chunks for the LLM
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    # Create the vector store and persist it to disk
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=str(CHROMA_PERSIST_DIR)
    )
    return vectorstore

# Global initialization so it only loads once per session
try:
    _vector_store = _get_or_create_vector_store()
except Exception as e:
    _vector_store = None
    print(f"Warning: Could not initialize RAG vector store. {e}")


@tool
def search_policy(query: str) -> str:
    """
    Perform a semantic search across all enterprise company policies, employee handbooks, and guidelines.
    
    Args:
        query: The user's question or search topic (e.g., 'What is the remote work policy?').
    """
    if not _vector_store:
        return "Error: The Knowledge Base is currently offline or uninitialized."
    
    # Retrieve the top 3 most relevant chunks from the PDFs
    results = _vector_store.similarity_search(query, k=3)
    
    if not results:
        return "No relevant information found in the company documents."
        
    # Format the retrieved chunks clearly for the agent to read
    formatted_results = []
    for doc in results:
        source_file = Path(doc.metadata.get('source', 'Unknown')).name
        page = doc.metadata.get('page', 'Unknown')
        formatted_results.append(f"--- Source: {source_file} (Page {page}) ---\n{doc.page_content}")
        
    return "\n\n".join(formatted_results)