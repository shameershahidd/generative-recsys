"""
Generative Recommendation System — Streamlit Demo
Run with: streamlit run app.py
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import load_dataset
from src.embeddings import build_item_embeddings, build_faiss_index, embed_text, get_model, retrieve_candidates
from src.baseline import get_user_embedding, recommend_dot_product
from src.verbalizer import verbalize_user_profile
from src.reranker import rerank_candidates


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Generative Recommender",
    page_icon="🎬",
    layout="wide",
)

# ── Results (update after full eval run) ─────────────────────────────────────

EVAL_RESULTS = {
    "n_users": 6034,
    "baseline":   {"NDCG@10": 0.0044, "HitRate@10": 0.0093, "Coverage": 0.2341},
    "generative": {"NDCG@10": 0.0155, "HitRate@10": 0.0288, "Coverage": 0.4437},
}

# ── Data + model loading (cached) ────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading dataset...")
def load_data():
    train_history, test_items, item_metadata, all_item_ids = load_dataset()
    return train_history, test_items, item_metadata, all_item_ids

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    return get_model()

@st.cache_resource(show_spinner="Building FAISS index...")
def load_index(_embed_model, _item_metadata, _all_item_ids):
    item_embeddings = build_item_embeddings(_item_metadata, _all_item_ids, _embed_model)
    faiss_index, id_to_idx, idx_to_id = build_faiss_index(item_embeddings, _all_item_ids)
    return item_embeddings, faiss_index, id_to_idx, idx_to_id

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🎬 Generative Recommendation System")
st.caption("LLM-verbalized user profiles vs. dot product baseline · MovieLens-1M")

# Sidebar — results table
with st.sidebar:
    st.header("📊 Evaluation Results")
    st.caption(f"Evaluated on {EVAL_RESULTS['n_users']} users · MovieLens-1M")

    b = EVAL_RESULTS["baseline"]
    g = EVAL_RESULTS["generative"]

    ndcg_lift = (g["NDCG@10"] - b["NDCG@10"]) / b["NDCG@10"] * 100
    hit_lift  = (g["HitRate@10"] - b["HitRate@10"]) / b["HitRate@10"] * 100
    cov_lift  = (g["Coverage"] - b["Coverage"]) / b["Coverage"] * 100

    col1, col2 = st.columns(2)
    col1.metric("NDCG@10 (baseline)", f"{b['NDCG@10']:.4f}")
    col2.metric("NDCG@10 (generative)", f"{g['NDCG@10']:.4f}", f"+{ndcg_lift:.0f}%")

    col1, col2 = st.columns(2)
    col1.metric("HitRate@10 (baseline)", f"{b['HitRate@10']:.3f}")
    col2.metric("HitRate@10 (generative)", f"{g['HitRate@10']:.3f}", f"+{hit_lift:.0f}%")

    col1, col2 = st.columns(2)
    col1.metric("Coverage (baseline)", f"{b['Coverage']:.3f}")
    col2.metric("Coverage (generative)", f"{g['Coverage']:.3f}", f"+{cov_lift:.0f}%")

    st.divider()
    st.markdown("**Stack**")
    st.markdown("- `sentence-transformers` — embeddings\n- `FAISS` — vector search\n- `Groq / Llama 3.1` — profile generation\n- `MovieLens-1M` — dataset")

# Main area — API key input
api_key = os.environ.get("GROQ_API_KEY", "")
if not api_key:
    api_key = st.text_input("Enter your Groq API key to enable recommendations", type="password")

if not api_key:
    st.info("Add your Groq API key above to try the live demo.")
    st.stop()

from groq import Groq
client = Groq(api_key=api_key)

# Load data
train_history, test_items, item_metadata, all_item_ids = load_data()
embed_model = load_embed_model()
item_embeddings, faiss_index, id_to_idx, idx_to_id = load_index(
    embed_model, item_metadata, all_item_ids
)

# User selection
st.subheader("Select a user")
col1, col2 = st.columns([3, 1])

user_ids = list(train_history.keys())
with col1:
    selected_user = st.selectbox("User ID", user_ids, index=0)
with col2:
    if st.button("🎲 Random user", use_container_width=True):
        selected_user = random.choice(user_ids)
        st.rerun()

# Show watch history
history = train_history[selected_user]
st.subheader("Watch history")
with st.expander(f"{len(history)} movies watched", expanded=True):
    cols = st.columns(3)
    for i, iid in enumerate(history[-12:]):  # show last 12
        meta = item_metadata.get(iid, {})
        cols[i % 3].markdown(f"**{meta.get('title', iid)}**  \n_{meta.get('genres', '')}_")

# Generate recommendations
if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
    exclude = set(history)

    col_baseline, col_generative = st.columns(2)

    # ── Baseline ─────────────────────────────────────────────────────────────
    with col_baseline:
        st.subheader("Dot Product Baseline")
        st.caption("Mean item embedding → cosine similarity ranking")

        user_vec = get_user_embedding(history, item_embeddings, id_to_idx)
        baseline_recs = recommend_dot_product(
            user_vec, item_embeddings, all_item_ids, top_k=10, exclude_ids=exclude
        )
        for rank, iid in enumerate(baseline_recs, 1):
            meta = item_metadata.get(iid, {})
            st.markdown(f"**{rank}.** {meta.get('title', iid)}  \n_{meta.get('genres', '')}_")

    # ── Generative ───────────────────────────────────────────────────────────
    with col_generative:
        st.subheader("Generative System")
        st.caption("LLM profile → semantic retrieval → LLM reranking")

        with st.spinner("Generating user profile..."):
            profile = verbalize_user_profile(history, item_metadata, client)

        st.info(f"**User profile:** {profile}")

        with st.spinner("Retrieving & reranking candidates..."):
            profile_vec = embed_text(profile, embed_model)
            candidates = retrieve_candidates(
                profile_vec, faiss_index, idx_to_id, top_k=20, exclude_ids=exclude
            )
            ranked = rerank_candidates(
                profile, candidates, item_metadata, client, top_k=10
            )

        for rank, (iid, reason) in enumerate(ranked, 1):
            meta = item_metadata.get(iid, {})
            st.markdown(
                f"**{rank}.** {meta.get('title', iid)}  \n"
                f"_{meta.get('genres', '')}_  \n"
                f"💬 {reason}"
            )
