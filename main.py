"""
Generative Recommendation System — main evaluation script.

Usage:
    # Baseline only (fast, no API calls):
    python main.py --mode baseline

    # Full generative system (requires ANTHROPIC_API_KEY):
    python main.py --mode generative

    # Both systems, side-by-side comparison:
    python main.py --mode both

    # Limit evaluation to N users (for quick testing):
    python main.py --mode both --n-users 100
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
import anthropic
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import load_dataset
from src.embeddings import build_item_embeddings, build_faiss_index, embed_text, get_model, retrieve_candidates
from src.baseline import get_user_embedding, recommend_dot_product
from src.verbalizer import verbalize_user_profile
from src.reranker import rerank_candidates
from src.evaluation import evaluate_recommender


def run_baseline(
    train_history: dict,
    test_items: dict,
    item_embeddings: np.ndarray,
    all_item_ids: list[int],
    id_to_idx: dict,
    n_users: int | None = None,
    top_k: int = 10,
) -> dict:
    print("\n=== Running Dot Product Baseline ===")
    user_ids = list(train_history.keys())
    if n_users:
        user_ids = user_ids[:n_users]

    recommendations = {}
    for uid in tqdm(user_ids, desc="Baseline"):
        history = train_history[uid]
        user_vec = get_user_embedding(history, item_embeddings, id_to_idx)
        exclude = set(history)
        recs = recommend_dot_product(
            user_vec, item_embeddings, all_item_ids, top_k=top_k, exclude_ids=exclude
        )
        recommendations[uid] = recs

    return recommendations


def run_generative(
    train_history: dict,
    test_items: dict,
    item_metadata: dict,
    all_item_ids: list[int],
    faiss_index,
    idx_to_id: dict,
    client: anthropic.Anthropic,
    model: SentenceTransformer,
    n_users: int | None = None,
    top_k: int = 10,
    retrieval_k: int = 50,
) -> dict:
    print("\n=== Running Generative System ===")
    user_ids = list(train_history.keys())
    if n_users:
        user_ids = user_ids[:n_users]

    recommendations = {}
    for uid in tqdm(user_ids, desc="Generative"):
        history = train_history[uid]
        exclude = set(history)

        # Step 1: Verbalize
        profile = verbalize_user_profile(history, item_metadata, client)

        # Step 2: Semantic retrieval from FAISS
        profile_vec = embed_text(profile, model)
        candidates = retrieve_candidates(
            profile_vec, faiss_index, idx_to_id, top_k=retrieval_k, exclude_ids=exclude
        )

        # Step 3: LLM reranking
        ranked = rerank_candidates(profile, candidates, item_metadata, client, top_k=top_k)
        recommendations[uid] = [iid for iid, _ in ranked]

    return recommendations


def print_results_table(results: dict[str, dict]):
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    headers = ["System", "NDCG@10", "Prec@10", "HitRate@10", "Coverage", "Users"]
    col_w = [20, 10, 10, 12, 10, 8]

    header_row = "".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header_row)
    print("-" * 60)

    for name, metrics in results.items():
        row = [
            name,
            f"{metrics.get('NDCG@10', 0):.4f}",
            f"{metrics.get('Precision@10', 0):.4f}",
            f"{metrics.get('HitRate@10', 0):.4f}",
            f"{metrics.get('Coverage', 0):.4f}",
            str(metrics.get("n_users", 0)),
        ]
        print("".join(v.ljust(w) for v, w in zip(row, col_w)))

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "generative", "both"], default="baseline")
    parser.add_argument("--n-users", type=int, default=None, help="Limit to N users for quick testing")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieval-k", type=int, default=50, help="FAISS candidates before LLM reranking")
    args = parser.parse_args()

    # Load data
    print("Loading dataset...")
    train_history, test_items, item_metadata, all_item_ids = load_dataset()
    print(f"  Users: {len(train_history):,}  |  Items: {len(all_item_ids):,}")

    # Build embeddings + FAISS index
    print("Building item embeddings...")
    embed_model = get_model()
    item_embeddings = build_item_embeddings(item_metadata, all_item_ids, embed_model)
    faiss_index, id_to_idx, idx_to_id = build_faiss_index(item_embeddings, all_item_ids)

    all_results = {}

    if args.mode in ("baseline", "both"):
        start = time.time()
        baseline_recs = run_baseline(
            train_history, test_items, item_embeddings, all_item_ids,
            id_to_idx, n_users=args.n_users, top_k=args.top_k,
        )
        elapsed = time.time() - start
        metrics = evaluate_recommender(baseline_recs, test_items, all_item_ids, k=args.top_k)
        metrics["time_s"] = elapsed
        all_results["Dot Product Baseline"] = metrics
        print(f"  Baseline done in {elapsed:.1f}s")

    if args.mode in ("generative", "both"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set. Export it and re-run.")
            sys.exit(1)

        client = anthropic.Anthropic(api_key=api_key)
        start = time.time()
        gen_recs = run_generative(
            train_history, test_items, item_metadata, all_item_ids,
            faiss_index, idx_to_id, client, embed_model,
            n_users=args.n_users, top_k=args.top_k, retrieval_k=args.retrieval_k,
        )
        elapsed = time.time() - start
        metrics = evaluate_recommender(gen_recs, test_items, all_item_ids, k=args.top_k)
        metrics["time_s"] = elapsed
        all_results["Generative (Claude)"] = metrics
        print(f"  Generative done in {elapsed:.1f}s")

    print_results_table(all_results)

    if "both" == args.mode and len(all_results) == 2:
        b = all_results["Dot Product Baseline"]
        g = all_results["Generative (Claude)"]
        ndcg_lift = (g["NDCG@10"] - b["NDCG@10"]) / b["NDCG@10"] * 100 if b["NDCG@10"] > 0 else 0
        cov_lift = (g["Coverage"] - b["Coverage"]) / b["Coverage"] * 100 if b["Coverage"] > 0 else 0
        print(f"\nNDCG@10 improvement:  {ndcg_lift:+.1f}%")
        print(f"Coverage improvement: {cov_lift:+.1f}%")


if __name__ == "__main__":
    main()
