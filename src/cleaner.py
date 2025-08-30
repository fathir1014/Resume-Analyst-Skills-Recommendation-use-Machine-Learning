# cleaner.py
"""
Text cleaning utilities for Resume Analyst project.
- HTML stripping, URL/email/number normalization
- Unicode normalization
- Stopword removal (English) with optional custom list
- Safe for pandas Series or raw strings
"""

from __future__ import annotations
from typing import Iterable, Optional
import re, html, unicodedata
import pandas as pd

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SK_STOPWORDS
    _SK_STOP = set(_SK_STOPWORDS)
except Exception:
    _SK_STOP = set()

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")

def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def _strip_html(text: str) -> str:
    # Remove tags and unescape entities
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    return text

def clean_text(
    text: str,
    lower: bool = True,
    strip_html: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_numbers: bool = True,
    remove_punct: bool = True,
    collapse_ws: bool = True,
    stopwords: Optional[Iterable[str]] = None,
    min_token_len: int = 2,
) -> str:
    """
    Clean a single text string.

    Returns
    -------
    str
        Cleaned text.
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    s = _normalize_unicode(text)

    if strip_html:
        s = _strip_html(s)

    if remove_urls:
        s = _URL_RE.sub(" ", s)

    if remove_emails:
        s = _EMAIL_RE.sub(" ", s)

    if lower:
        s = s.lower()

    if remove_numbers:
        s = _NUM_RE.sub(" ", s)

    if remove_punct:
        s = _NON_ALNUM_RE.sub(" ", s)

    # Collapse whitespace
    if collapse_ws:
        s = _MULTI_WS_RE.sub(" ", s).strip()

    # Stopword filtering
    if stopwords is None:
        stop_set = _SK_STOP
    else:
        stop_set = set(stopwords)

    if stop_set:
        tokens = [t for t in s.split() if (len(t) >= min_token_len and t not in stop_set)]
        s = " ".join(tokens)

    return s

def clean_series(
    series: pd.Series,
    **kwargs
) -> pd.Series:
    """
    Vectorized clean_text over a pandas Series.
    """
    return series.astype(str).map(lambda x: clean_text(x, **kwargs))

def default_cleaning(series: pd.Series) -> pd.Series:
    """
    Opinionated default cleaner tuned for resumes.
    """
    return clean_series(series,
                        lower=True,
                        strip_html=True,
                        remove_urls=True,
                        remove_emails=True,
                        remove_numbers=True,
                        remove_punct=True,
                        collapse_ws=True,
                        stopwords=None,  # use sklearn default
                        min_token_len=2)
