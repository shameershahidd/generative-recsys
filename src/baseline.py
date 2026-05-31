"""
Dot product baseline recommender.
User vector = mean of interacted item embeddings.
Ranking = dot product (cosine sim) against all candidate items.
"""

import numpy as np


def get_user_embedding(user_history: list[int], item_embeddings: np.ndarray, id_to_idx: dict) -> np.ndarray:
    vecs = [item_embeddings[id_to_idx[iid]] for iid in user_history if iid in id_to_idx]
    if not vecs:
        return np.zeros(item_embeddings.shape[1])
    user_vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm
    return user_vec


def recommend_dot_product(
    user_vec: np.ndarray,
    item_embeddings: np.ndarray,
    all_item_ids: list[int],
    top_k: int = 10,
    exclude_ids: set | None = None,
) -> list[int]:
    scores = item_embeddings @ user_vec
    sorted_indices = np.argsort(scores)[::-1]

    results = []
    for idx in sorted_indices:
        iid = all_item_ids[idx]
        if exclude_ids and iid in exclude_ids:
            continue
        results.append(iid)
        if len(results) >= top_k:
            break
    return results
