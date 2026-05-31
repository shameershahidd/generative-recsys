"""
Evaluation metrics for recommender systems.
NDCG@K, Precision@K, and catalog Coverage.
"""

import numpy as np
from collections import defaultdict


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    recommended_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended_k)
        if item in relevant
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / k


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    return float(any(item in relevant for item in recommended[:k]))


def catalog_coverage(all_recommendations: list[list[int]], total_items: int) -> float:
    """Fraction of the item catalog that appears in any recommendation list."""
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)
    return len(recommended_items) / total_items if total_items > 0 else 0.0


def evaluate_recommender(
    recommendations: dict[int, list[int]],
    test_items: dict[int, list[int]],
    all_item_ids: list[int],
    k: int = 10,
) -> dict[str, float]:
    """
    Computes aggregate metrics across all users.

    Args:
        recommendations: {user_id: [ranked movie_ids]}
        test_items:      {user_id: [relevant movie_ids]}
        all_item_ids:    full catalog for coverage computation
        k:               cutoff for NDCG and Precision

    Returns:
        dict with mean NDCG@k, Precision@k, HitRate@k, Coverage
    """
    ndcgs, precisions, hits = [], [], []
    all_recs = []

    for user_id, recs in recommendations.items():
        if user_id not in test_items:
            continue
        relevant = set(test_items[user_id])
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        precisions.append(precision_at_k(recs, relevant, k))
        hits.append(hit_rate_at_k(recs, relevant, k))
        all_recs.append(recs[:k])

    coverage = catalog_coverage(all_recs, len(all_item_ids))

    return {
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"Precision@{k}": float(np.mean(precisions)),
        f"HitRate@{k}": float(np.mean(hits)),
        "Coverage": coverage,
        "n_users": len(ndcgs),
    }
