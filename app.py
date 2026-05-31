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

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Generative Recommender",
    page_icon="🎬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Hide default header padding */
.block-container { padding-top: 2rem; }

/* Movie card */
.movie-card {
    background: #1e1e2e;
    border: 1px solid #2e2e4e;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.movie-card:hover { border-color: #5b5bff; }
.movie-rank {
    font-size: 11px;
    font-weight: 700;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.movie-title {
    font-size: 15px;
    font-weight: 600;
    color: #e8e8f0;
    margin: 2px 0 4px 0;
}
.genre-badge {
    display: inline-block;
    background: #2e2e4e;
    color: #a0a0c0;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 20px;
    margin: 2px 3px 2px 0;
}
.reason-text {
    font-size: 12px;
    color: #7c7caa;
    margin-top: 6px;
    font-style: italic;
}
.profile-box {
    background: linear-gradient(135deg, #1a1a3e, #2a1a4e);
    border: 1px solid #4a3a8e;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 14px;
    color: #c8b8f8;
    line-height: 1.6;
}
.metric-label {
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.section-header {
    font-size: 13px;
    font-weight: 700;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2e2e4e;
}
</style>
""", unsafe_allow_html=True)

# ── Eval results ──────────────────────────────────────────────────────────────

EVAL_RESULTS = {
    "n_users": 6034,
    "baseline":   {"NDCG@10": 0.0044, "Precision@10": 0.0009, "HitRate@10": 0.0093, "Coverage": 0.2341},
    "generative": {"NDCG@10": 0.0155, "Precision@10": 0.0029, "HitRate@10": 0.0288, "Coverage": 0.4437},
}

# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading dataset...")
def load_data():
    return load_dataset()

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    return get_model()

@st.cache_resource(show_spinner="Building FAISS index...")
def load_index(_embed_model, _item_metadata, _all_item_ids):
    item_embeddings = build_item_embeddings(_item_metadata, _all_item_ids, _embed_model)
    faiss_index, id_to_idx, idx_to_id = build_faiss_index(item_embeddings, _all_item_ids)
    return item_embeddings, faiss_index, id_to_idx, idx_to_id

# ── Helpers ───────────────────────────────────────────────────────────────────

def genre_badges(genres_str: str) -> str:
    badges = "".join(
        f'<span class="genre-badge">{g.strip()}</span>'
        for g in genres_str.split(",") if g.strip()
    )
    return badges

def movie_card(rank: int, iid: int, item_metadata: dict, reason: str = "") -> str:
    meta = item_metadata.get(iid, {})
    title = meta.get("title", str(iid))
    genres = meta.get("genres", "")
    badges = genre_badges(genres)
    reason_html = f'<div class="reason-text">💬 {reason}</div>' if reason else ""
    return f"""
    <div class="movie-card">
        <div class="movie-rank">#{rank}</div>
        <div class="movie-title">{title}</div>
        {badges}
        {reason_html}
    </div>
    """

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎬 Generative RecSys")
    st.caption("MovieLens-1M · LLM-augmented recommendations")

    st.divider()

    st.markdown('<div class="section-header">Evaluation Results</div>', unsafe_allow_html=True)
    st.caption(f"All {EVAL_RESULTS['n_users']:,} users · leave-last-out split")

    b = EVAL_RESULTS["baseline"]
    g = EVAL_RESULTS["generative"]

    metrics = ["NDCG@10", "HitRate@10", "Coverage"]
    labels  = ["NDCG@10", "HitRate@10", "Coverage"]

    for metric, label in zip(metrics, labels):
        bv, gv = b[metric], g[metric]
        lift = (gv - bv) / bv * 100
        col1, col2 = st.columns(2)
        col1.metric(f"Baseline {label}", f"{bv:.4f}")
        col2.metric(f"Generative {label}", f"{gv:.4f}", f"+{lift:.0f}%")

    st.divider()

    import pandas as pd
    chart_data = pd.DataFrame({
        "Baseline":   [b["NDCG@10"], b["HitRate@10"], b["Coverage"]],
        "Generative": [g["NDCG@10"], g["HitRate@10"], g["Coverage"]],
    }, index=["NDCG@10", "HitRate@10", "Coverage"])
    st.bar_chart(chart_data, color=["#7fb3d3", "#2e86ff"])

    st.divider()
    st.markdown("**Stack**")
    st.markdown(
        "- `sentence-transformers` — MiniLM-L6-v2\n"
        "- `FAISS` — vector search\n"
        "- `Groq / Llama-3.1-8B` — profile + reranking\n"
        "- MovieLens-1M — 6,034 users · 3,883 movies"
    )

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("# 🎬 Generative Recommendation System")
st.markdown(
    "Pick a user to see their watch history, then compare a **dot product baseline** "
    "against an **LLM-powered pipeline** that builds a natural language taste profile "
    "before retrieving candidates."
)

st.divider()

# API key
api_key = os.environ.get("GROQ_API_KEY", "")
if not api_key:
    api_key = st.text_input("Groq API key (required for generative recommendations)", type="password")
if not api_key:
    st.info("Enter your Groq API key above to enable the demo. Get one free at console.groq.com")
    st.stop()

from groq import Groq
client = Groq(api_key=api_key)

train_history, test_items, item_metadata, all_item_ids = load_data()
embed_model = load_embed_model()
item_embeddings, faiss_index, id_to_idx, idx_to_id = load_index(embed_model, item_metadata, all_item_ids)

# User picker
st.markdown('<div class="section-header">Select a User</div>', unsafe_allow_html=True)
user_ids = list(train_history.keys())

col1, col2 = st.columns([4, 1])
with col1:
    selected_user = st.selectbox("User ID", user_ids, index=0, label_visibility="collapsed")
with col2:
    if st.button("🎲 Random", use_container_width=True):
        st.session_state["random_user"] = random.choice(user_ids)
        st.rerun()

if "random_user" in st.session_state:
    selected_user = st.session_state.pop("random_user")

history = train_history[selected_user]

# Watch history
st.markdown(f'<div class="section-header">Watch History — {len(history)} movies</div>', unsafe_allow_html=True)
with st.expander("Show watch history", expanded=False):
    cols = st.columns(3)
    for i, iid in enumerate(history[-12:]):
        meta = item_metadata.get(iid, {})
        with cols[i % 3]:
            st.markdown(
                f"**{meta.get('title', iid)}**  \n"
                f"<span style='color:#888;font-size:12px'>{meta.get('genres','')}</span>",
                unsafe_allow_html=True
            )

st.divider()

# Generate button
if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
    exclude = set(history)

    col_baseline, col_generative = st.columns(2)

    with col_baseline:
        st.markdown('<div class="section-header">Dot Product Baseline</div>', unsafe_allow_html=True)
        st.caption("Mean item embedding → cosine similarity ranking")

        user_vec = get_user_embedding(history, item_embeddings, id_to_idx)
        baseline_recs = recommend_dot_product(
            user_vec, item_embeddings, all_item_ids, top_k=10, exclude_ids=exclude
        )
        cards_html = "".join(movie_card(i + 1, iid, item_metadata) for i, iid in enumerate(baseline_recs))
        st.markdown(cards_html, unsafe_allow_html=True)

    with col_generative:
        st.markdown('<div class="section-header">Generative System</div>', unsafe_allow_html=True)
        st.caption("LLM taste profile → semantic retrieval → LLM reranking")

        with st.spinner("Generating taste profile..."):
            profile = verbalize_user_profile(history, item_metadata, client)

        st.markdown(f'<div class="profile-box">✨ <strong>Taste profile:</strong> {profile}</div>', unsafe_allow_html=True)

        with st.spinner("Retrieving & reranking..."):
            profile_vec = embed_text(profile, embed_model)
            candidates = retrieve_candidates(
                profile_vec, faiss_index, idx_to_id, top_k=20, exclude_ids=exclude
            )
            ranked = rerank_candidates(profile, candidates, item_metadata, client, top_k=10)

        cards_html = "".join(
            movie_card(i + 1, iid, item_metadata, reason)
            for i, (iid, reason) in enumerate(ranked)
        )
        st.markdown(cards_html, unsafe_allow_html=True)
