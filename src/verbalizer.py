"""
LLM-based user profile verbalizer.
Converts interaction history into a natural language taste profile.
"""

import re
import time
from groq import Groq, RateLimitError


def verbalize_user_profile(
    user_history: list[int],
    item_metadata: dict,
    client: Groq,
    model: str = "llama-3.1-8b-instant",
    max_history: int = 15,
    max_retries: int = 5,
) -> str:
    history = user_history[-max_history:]
    history_lines = [
        f"- {item_metadata[iid]['title']} ({item_metadata[iid]['genres']})"
        for iid in history
        if iid in item_metadata
    ]
    history_text = "\n".join(history_lines)

    prompt = f"""Movies this user has watched:
{history_text}

Write 2 sentences describing their taste and what they'd enjoy next."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            wait = _parse_wait_seconds(str(e)) + 1
            time.sleep(wait)
        except Exception:
            return history_text  # Fallback: use raw history as profile

    return history_text


def _parse_wait_seconds(error_msg: str) -> float:
    match = re.search(r"try again in ([\d.]+)s", error_msg)
    return float(match.group(1)) if match else 10.0
