"""LLM summarization with provider fallback (Groq -> Anthropic -> truncation)."""

import logging

logger = logging.getLogger(__name__)

_REFUSAL_PATTERNS = [
    "no text provided",
    "i don't see any text",
    "please provide the text",
    "you haven't provided",
    "no text to summarize",
    "provide the text you",
    "i cannot summarize",
    "there is no text",
]


def is_usable_summary(response_text: str) -> bool:
    if not response_text or len(response_text) < 10:
        return False
    lower = response_text.lower()
    return not any(pat in lower for pat in _REFUSAL_PATTERNS)


def summarize(text: str) -> str:
    prompt = f"Summarize the following text in 2-3 sentences. Focus on the key points.\n\nText:\n{text}\n\nSummary:"

    # 1) Try Groq (free tier)
    try:
        from groq import Groq
        groq_client = Groq()
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.choices[0].message.content.strip()
        if is_usable_summary(result):
            logger.info("Summarization via groq")
            return result
        logger.warning(f"Groq returned unusable summary, falling through: {result!r}")
    except Exception as e:
        logger.warning(f"Groq summarization failed: {e}")

    # 2) Fall back to Anthropic (paid, deployer's own key)
    try:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        result = message.content[0].text.strip()
        if is_usable_summary(result):
            logger.info("Summarization via anthropic (fallback)")
            return result
        logger.warning(f"Anthropic returned unusable summary, falling through: {result!r}")
    except Exception as e:
        logger.warning(f"Anthropic summarization failed: {e}")

    # 3) Last resort: truncation
    logger.warning("Summarization via truncation (all providers failed or returned unusable output)")
    return text[:600]
