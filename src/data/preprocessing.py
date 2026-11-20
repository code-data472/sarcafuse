"""Text preprocessing for social media input."""
import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_SARCASM_TAGS = {"#sarcasm", "#irony", "#sarcastic", "#not"}

def preprocess_tweet(text, remove_urls=True, remove_mentions=True, expand_hashtags=True, reduce_repeats=True, remove_sarcasm_tags=True):
    if remove_urls:
        text = _URL_RE.sub(" ", text)
    if remove_mentions:
        text = _MENTION_RE.sub(" ", text)
    if remove_sarcasm_tags:
        for tag in _SARCASM_TAGS:
            text = text.replace(tag, " ")
    if expand_hashtags:
        text = _HASHTAG_RE.sub(r"\1", text)
    if reduce_repeats:
        text = _REPEAT_RE.sub(r"\1\1", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_tokens(tokens):
    return [t.lower() for t in tokens if t.strip()]
