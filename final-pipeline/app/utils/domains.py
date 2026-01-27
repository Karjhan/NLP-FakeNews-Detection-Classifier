from __future__ import annotations

from urllib.parse import urlparse


def get_domain(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return ""


def normalize_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    return d.replace("www.", "")


def is_platform_domain(domain: str, platform_set: set[str]) -> bool:
    d = normalize_domain(domain)
    return d in platform_set
