"""
LLM reranking layer.
Takes top-K FAISS candidates + user profile and reranks with explanations.
"""

import json
import re
import anthropic


RERANK_PROMPT = """\
You are a movie recommendation expert.

User profile:
{user_profile}

Candidate movies to rerank (numbered list):
{candidates_text}

Rerank all {n} candidates from most to least relevant for this user.
Return ONLY a JSON array with objects in this exact format:
[
  {{"rank": 1, "item_number": <original number>, "reason": "<one sentence>"}},
  ...
]
All {n} candidates must appear exactly once. No extra text outside the JSON."""


def rerank_candidates(
    user_profile: str,
    candidates: list[int],
    item_metadata: dict,
    client: anthropic.Anthropic,
    model: str = "claude-haiku-4-5-20251001",
    top_k: int = 10,
) -> list[tuple[int, str]]:
    """
    Returns top_k (movie_id, reason) tuples ordered by LLM relevance.
    Falls back to original candidate order if parsing fails.
    """
    if not candidates:
        return []

    candidates_lines = []
    for i, iid in enumerate(candidates, 1):
        meta = item_metadata.get(iid, {})
        title = meta.get("title", f"Item {iid}")
        desc = meta.get("description", title)
        candidates_lines.append(f"{i}. {desc}")

    candidates_text = "\n".join(candidates_lines)
    n = len(candidates)

    prompt = RERANK_PROMPT.format(
        user_profile=user_profile,
        candidates_text=candidates_text,
        n=n,
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    try:
        ranked = _parse_rerank_response(raw, candidates)
        return ranked[:top_k]
    except Exception:
        # Fallback: return candidates in original order with no reason
        return [(iid, "") for iid in candidates[:top_k]]


def _parse_rerank_response(
    raw: str, candidates: list[int]
) -> list[tuple[int, str]]:
    # Extract JSON array even if surrounded by markdown fences
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in response")

    data = json.loads(match.group())
    results = []
    for entry in data:
        item_num = int(entry["item_number"])
        reason = entry.get("reason", "")
        if 1 <= item_num <= len(candidates):
            results.append((candidates[item_num - 1], reason))

    return results
