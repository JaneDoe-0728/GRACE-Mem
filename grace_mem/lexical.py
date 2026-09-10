"""Lexical primitives both capabilities share.

BM25 tokenization is needed on both sides of the system: ingestion tokenizes
what it stores, retrieval tokenizes what it searches for, and the two must
agree token for token or the sparse index silently misses. It used to live in
`ingestion/parsing.py`, which meant retrieval imported ingestion for it --
a dependency between two capabilities that are otherwise independent, and one
the architecture documents said did not exist.

Nothing here knows about entities, evidence, or either pipeline. It is text in,
tokens out, which is why it can sit below both.
"""

import re

# Prefer NLTK's word_tokenize; fall back to a regex if the resources or the
# install are missing
from nltk import pos_tag, word_tokenize

# Extend as needed
EN_STOPWORDS = {
    "the","a","an","in","on","at","for","to","from","of",
    "and","or","but","with","without","is","are","was","were",
    "be","been","being","it","its","they","them","this","that",
    "these","those","as","by","into","over","under","up","down",
}

# regex fallback
_WORD_RE = re.compile(r"[a-zA-Z]+")

# ===== date token patterns =====

# 1) ISO / full year formats
_DATE_YMD_RE = re.compile(
    r"^\d{4}([-/.])\d{1,2}\1\d{1,2}$" # matches: 2023-03-04, 2023/3/4, 2023.03.04
)

# 2) Short numeric dates (month/day)
_DATE_MD_RE = re.compile(
    r"^\d{1,2}/\d{1,2}$" # matches: 4/1, 04/01
)

def is_date_token(token: str) -> bool:
    """Return whether a token already looks like a date literal."""
    if not token:
        return False
    return bool(
        _DATE_YMD_RE.match(token)
        or _DATE_MD_RE.match(token)
    )

def tokenize_en(text: str) -> list[str]:
    """Tokenize English text for BM25 while preserving informative date tokens."""
    t = (text or "").strip()
    if not t:
        return []

    # Step 1: basic tokenize
    try:
        toks = word_tokenize(t)
    except Exception:
        toks = _WORD_RE.findall(t)
        
    toks = [w.lower() for w in toks if any(c.isalpha() for c in w) or is_date_token(w)]

    # Step 2: POS tagging
    try:
        tags = pos_tag(toks)
    except Exception:
        # Fallback: with no POS tagger, keep every token (nltk is installed here,
        # so this is effectively unreachable)
        return toks

    keep_tags = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "FW"}

    filtered = []

    for token, tag in tags:
        # Dates are kept verbatim, bypassing the POS, stopword and length rules
        if is_date_token(token):
            filtered.append(token)
            continue

        # Length too short
        if len(token) < 3:
            continue

        # Stopwords / known noise
        if token in EN_STOPWORDS:
            continue

        # POS-based filtering
        if tag in keep_tags:
            filtered.append(token)
            continue

        # Fallback for unknown proper nouns / foreign words
        # 1) all-alphabetic & length >= 3 & not a stopword -> keep
        if token.isalpha():
            filtered.append(token)
            continue

    return filtered
