from __future__ import annotations

import re
from typing import Optional


def normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def safe_str(x) -> str:
    if x is None:
        return ""
    return str(x)


def build_text_input(
    *,
    title: Optional[str] = None,
    claim: Optional[str] = None,
    body: Optional[str] = None,
    prefer_body_min_len: int = 200,
) -> str:
    title = normalize_ws(safe_str(title))
    claim = normalize_ws(safe_str(claim))
    body = normalize_ws(safe_str(body))

    short = ""
    if title and claim:
        short = f"{title} [SEP] {claim}"
    elif title:
        short = title
    elif claim:
        short = claim

    if body and len(body) >= prefer_body_min_len:
        if short:
            return f"[SHORT] {short}\n[LONG] {body}".strip()
        return body.strip()

    return short.strip() or body.strip()


def text_len(s: str) -> int:
    return len((s or "").strip())


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def strip_urls(text: str) -> str:
    return _URL_RE.sub(" ", text or "")
