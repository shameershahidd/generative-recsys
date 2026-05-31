"""
LLM reranking layer.
Takes top-K FAISS candidates + user profile and reranks with explanations.
"""

import json
import re
import time
from groq import Groq, RateLimitError


RERANK_PROMPT = """\
You are a movie recommendation expert.

User profile:
{user_profile}

Candidate movies (numbered):
{candidates_text}

Rerank all {n} candidates from most to least relevant for this user.
Return ONLY a JSON array:
[{{"rank": 1, "item_number": <number>, "reason": "<one sentence>"}}, ...]
All {n} items must appear. No extra text."""


def rerank_candidates(
    user_profile: str,
    candidates: list[int],
    item_metadata: dict,
    client: Groq,
    model: str = "llama-3.1-8b-instant",
    top_k: int = 10,
    max_retries: int = 5,
) -> list[tuple[int, str]]:
    """
    Returns top_k (movie_id, reason) tuples ordered by LLM relevance.
    Falls back to original candidate order if parsing fails.
    """
    if not candidates:
        return []

    # Use only titles to keep prompt short and stay within rate limits
    candidates_lines = [
        f"{i+1}. {item_metadata.get(iid, {}).get('title', f'Item {iid}')}"
        for i, iid in enumerate(candidates)
    ]
    candidates_text = "\n".join(candidates_lines)
    n = len(candidates)

    prompt = RERANK_PROMPT.format(
        user_profile=user_profile,
        candidates_text=candidates_text,
        n=n,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()
            ranked = _parse_rerank_response(raw, candidates)
            return ranked[:top_k]
        except RateLimitError as e:
            wait = _parse_wait_seconds(str(e)) + 1
            time.sleep(wait)
        except Exception:
            return [(iid, "") for iid in candidates[:top_k]]

    return [(iid, "") for iid in candidates[:top_k]]


def _parse_wait_seconds(error_msg: str) -> float:
    match = re.search(r"try again in ([\d.]+)s", error_msg)
    return float(match.group(1)) if match else 10.0


def _parse_rerank_response(raw: str, candidates: list[int]) -> list[tuple[int, str]]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found")
    data = json.loads(match.group())
    results = []
    for entry in data:
        item_num = int(entry["item_number"])
        reason = entry.get("reason", "")
        if 1 <= item_num <= len(candidates):
            results.append((candidates[item_num - 1], reason))
    return results
