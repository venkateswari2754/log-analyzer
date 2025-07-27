import os
import pickle
import pandas as pd
from langchain_community.vectorstores import FAISS

from app.ai.embeddings import get_embeddings_model

# Directory for storing cached FAISS indexes and DataFrames
CACHE_DIR = "log_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _get_file_path(log_hash: str, extension: str) -> str:
    return os.path.join(CACHE_DIR, f"{log_hash}.{extension}")

# Save vector store to disk
def save_vector_store(log_hash: str, vector_store: FAISS):
    path = _get_file_path(log_hash, "faiss")
    vector_store.save_local(path)

# Load vector store from disk
def load_vector_store(log_hash: str) -> FAISS | None:
    path = _get_file_path(log_hash, "faiss")
    if os.path.exists(path):
        try:
            embeddings = get_embeddings_model()
            return FAISS.load_local(path, embeddings)
        except Exception as e:
            print(f"[load_vector_store] Error: {e}")
    return None

# Save parsed log DataFrame
def save_cached_data(log_hash: str, df: pd.DataFrame):
    path = _get_file_path(log_hash, "pkl")
    try:
        with open(path, "wb") as f:
            pickle.dump(df, f)
    except Exception as e:
        print(f"[save_cached_data] Error: {e}")

# Load parsed log DataFrame
def load_cached_data(log_hash: str) -> pd.DataFrame | None:
    path = _get_file_path(log_hash, "pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[load_cached_data] Error: {e}")
    return None
