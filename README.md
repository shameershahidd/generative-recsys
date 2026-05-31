# Generative Recommendation System

Comparing a dot-product content-based baseline against an LLM-augmented pipeline on MovieLens-1M. The generative system converts each user's watch history into a natural language taste profile, retrieves candidates via semantic search, and ranks them — achieving **+249.8% NDCG@10** over the baseline.

## Results

| System | NDCG@10 | Precision@10 | HitRate@10 | Coverage |
|---|---|---|---|---|
| Dot Product Baseline | 0.0044 | 0.0009 | 0.0093 | 0.2341 |
| Generative (Groq/Llama-3.1-8B) | 0.0155 | 0.0029 | 0.0288 | 0.4437 |
| **Improvement** | **+249.8%** | **+222%** | **+209.7%** | **+89.5%** |

Evaluated on all 6,034 users from MovieLens-1M using leave-last-out splits.

LLM reranking is disabled in the numbers above (free-tier rate limits make it impractical at 6K users). On a 50-user subset with `--rerank` enabled, it provides an additional lift on top of semantic retrieval alone.

## Architecture

```
User history → [Verbalizer] → natural language profile
                                        ↓
                              [FAISS semantic retrieval]  ← item embeddings (MiniLM-L6)
                                        ↓
                                top-20 candidates
                                        ↓
                               [LLM reranking] (optional)
                                        ↓
                               final top-10 recommendations
```

**Baseline:** averages item embeddings from a user's watch history, ranks all items by dot product similarity.

**Generative pipeline:**
1. **Verbalize** — Llama-3.1-8B summarizes the user's watch history into a 2-sentence taste profile
2. **Retrieve** — the profile is embedded with `all-MiniLM-L6-v2` and queried against a FAISS index of all 3,883 movie embeddings
3. **Rerank** — optional second LLM pass to reorder the top-20 candidates with explanations (`--rerank` flag); disabled by default due to free-tier rate limits at scale

## Demo

A Streamlit app (`app.py`) lets you pick any user, see their watch history, and compare recommendations side-by-side in real time.

```bash
export GROQ_API_KEY=your_key_here
streamlit run app.py
```

Or deploy to [Streamlit Cloud](https://streamlit.io/cloud) by connecting this repo — set `GROQ_API_KEY` as a secret in the app settings.

## Notebook

`notebooks/analysis.ipynb` walks through dataset stats, metric visualizations, and a live side-by-side recommendation example for any user.

## Stack

- `sentence-transformers` — `all-MiniLM-L6-v2` for item and profile embeddings
- `faiss-cpu` — approximate nearest neighbor retrieval
- `groq` — Llama-3.1-8B-Instant for verbalization and reranking
- MovieLens-1M — 1M ratings from 6,034 users on 3,883 movies

## Setup

```bash
git clone <repo-url>
cd generative-recsys
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## Usage

```bash
# Baseline only (no API key needed)
python main.py --mode baseline

# Full generative pipeline
export GROQ_API_KEY=your_key_here
python main.py --mode generative

# Side-by-side comparison (reproduces the results table above)
python main.py --mode both

# Quick test on 100 users
python main.py --mode both --n-users 100

# Enable LLM reranking on a small subset (hits rate limits at full scale)
python main.py --mode generative --n-users 50 --rerank
```

Item embeddings and the FAISS index are cached to `data/cache/` after the first run.

## Project structure

```
src/
  data_pipeline.py   — load MovieLens-1M, leave-last-out split
  embeddings.py      — MiniLM embeddings, FAISS index build + query
  baseline.py        — dot product recommender
  verbalizer.py      — LLM taste profile generation
  reranker.py        — LLM candidate reranking
  evaluation.py      — NDCG@K, Precision@K, HitRate@K, Coverage
main.py              — CLI entry point
app.py               — Streamlit demo
notebooks/
  analysis.ipynb     — dataset EDA, metric plots, example recommendations
```
