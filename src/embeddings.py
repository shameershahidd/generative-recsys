"""
Item embedding generation and management using sentence-transformers.
Builds and caches FAISS index for fast retrieval.
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
MODEL_NAME = "all-MiniLM-L6-v2"


def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def build_item_embeddings(
    item_metadata: dict,
    all_item_ids: list[int],
    model: SentenceTransformer | None = None,
    cache: bool = True,
) -> np.ndarray:
    """
    Embeds all items using their description text.
    Returns array of shape (n_items, embed_dim) aligned with all_item_ids order.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "item_embeddings.pkl"

    if cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached["item_ids"] == all_item_ids:
            print("Loaded item embeddings from cache.")
            return cached["embeddings"]

    if model is None:
        model = get_model()

    texts = [item_metadata[iid]["description"] for iid in all_item_ids]
    print(f"Embedding {len(texts)} items...")
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump({"item_ids": all_item_ids, "embeddings": embeddings}, f)

    return embeddings


def embed_text(text: str, model: SentenceTransformer | None = None) -> np.ndarray:
    """Embed a single text string (e.g., a user profile)."""
    if model is None:
        model = get_model()
    vec = model.encode([text], normalize_embeddings=True)
    return vec[0]


def build_faiss_index(
    item_embeddings: np.ndarray,
    all_item_ids: list[int],
    cache: bool = True,
) -> faiss.IndexFlatIP:
    """
    Builds a flat inner-product FAISS index (works for cosine sim with normalized vecs).
    Returns (index, id_to_idx, idx_to_id) mapping.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index_path = CACHE_DIR / "faiss.index"
    map_path = CACHE_DIR / "faiss_maps.pkl"

    if cache and index_path.exists() and map_path.exists():
        print("Loaded FAISS index from cache.")
        index = faiss.read_index(str(index_path))
        with open(map_path, "rb") as f:
            maps = pickle.load(f)
        return index, maps["id_to_idx"], maps["idx_to_id"]

    dim = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(item_embeddings.astype(np.float32))

    idx_to_id = {i: iid for i, iid in enumerate(all_item_ids)}
    id_to_idx = {iid: i for i, iid in enumerate(all_item_ids)}

    if cache:
        faiss.write_index(index, str(index_path))
        with open(map_path, "wb") as f:
            pickle.dump({"idx_to_id": idx_to_id, "id_to_idx": id_to_idx}, f)

    return index, id_to_idx, idx_to_id


def retrieve_candidates(
    query_vec: np.ndarray,
    index: faiss.IndexFlatIP,
    idx_to_id: dict,
    top_k: int = 50,
    exclude_ids: set | None = None,
) -> list[int]:
    """
    Returns top_k movie_ids from FAISS search, excluding any in exclude_ids.
    Retrieves extra items to account for filtering.
    """
    fetch_k = top_k + (len(exclude_ids) if exclude_ids else 0) + 10
    fetch_k = min(fetch_k, index.ntotal)

    scores, indices = index.search(query_vec.reshape(1, -1).astype(np.float32), fetch_k)
    results = []
    for idx in indices[0]:
        if idx < 0:
            continue
        iid = idx_to_id[idx]
        if exclude_ids and iid in exclude_ids:
            continue
        results.append(iid)
        if len(results) >= top_k:
            break
    return results
