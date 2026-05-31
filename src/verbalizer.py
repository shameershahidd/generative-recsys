"""
LLM-based user profile verbalizer.
Converts interaction history into a natural language taste profile.
"""

import anthropic


def verbalize_user_profile(
    user_history: list[int],
    item_metadata: dict,
    client: anthropic.Anthropic,
    model: str = "claude-haiku-4-5-20251001",
    max_history: int = 20,
) -> str:
    """
    Returns a 2-3 sentence natural language profile describing the user's taste.
    Uses the most recent max_history items to keep the prompt tight.
    """
    history = user_history[-max_history:]
    history_lines = []
    for iid in history:
        if iid in item_metadata:
            meta = item_metadata[iid]
            history_lines.append(f"- {meta['title']} ({meta['genres']})")

    history_text = "\n".join(history_lines)

    prompt = f"""A user has interacted with the following movies (most recent last):
{history_text}

Write a concise natural language profile describing this user's tastes, preferences, \
and what kinds of movies they are likely looking for next. \
Be specific about themes, genres, and narrative styles. \
2-3 sentences maximum."""

    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
