from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
import re
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
from tqdm import tqdm

# -----------------------
# Config
# -----------------------

BASE = "https://verificat.afp.com"
LISTING_URL = f"{BASE}/list/Romania"

OUT_DIR = Path("out_afp_verificat")
OUT_DIR.mkdir(exist_ok=True)

MAX_LISTING_CLICKS = 40
REQUEST_DELAY = 1.0
HEADLESS = False

DOC_URL_RE = re.compile(r"^https://verificat\.afp\.com/doc\.afp\.com\.\w+$")

# -----------------------
# Utilities
# -----------------------

def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    return " ".join(str(label).strip().split()).upper()

FALSE_SET = {
    "FALS",
    "ÎNȘELĂTOR",
    "LIPSA CONTEXTULUI",
    "FOTOGRAFIE ALTERATĂ",
    "VIDEOCLIP ALTERAT",
    "SATIRĂ",
    "FARSĂ",
    "DEEPFAKE",
}
TRUE_SET = {"ADEVĂRAT"}

def label_group(label: Optional[str]) -> Optional[str]:
    nl = normalize_label(label)
    if nl in TRUE_SET:
        return "TRUE"
    if nl in FALSE_SET:
        return "FALSE"
    return "OTHER" if nl else None

# -----------------------
# JSON-LD parsing
# -----------------------

def find_claimreview(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        if obj.get("@type") == "ClaimReview":
            return obj
        if "claimReviewed" in obj and "reviewRating" in obj:
            return obj
        if "@graph" in obj:
            for n in obj["@graph"]:
                hit = find_claimreview(n)
                if hit:
                    return hit
        for v in obj.values():
            hit = find_claimreview(v)
            if hit:
                return hit
    elif isinstance(obj, list):
        for it in obj:
            hit = find_claimreview(it)
            if hit:
                return hit
    return None

def parse_article(html: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    jsonlds = []
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            jsonlds.append(json.loads(s.string))
        except Exception:
            pass

    cr = None
    for block in jsonlds:
        cr = find_claimreview(block)
        if cr:
            break

    rec = {
        "id": sha1(url),
        "url": url,
        "type": "afp_factcheck",
        "language": "ro",
        "claim": None,
        "label": None,
        "date_verified": None,
        "speaker": None,
        "source_url": None,
        "title": None,
    }

    if cr:
        rr = cr.get("reviewRating", {})
        item = cr.get("itemReviewed", {})
        author = item.get("author", {})

        rec.update({
            "claim": cr.get("claimReviewed"),
            "label": rr.get("alternateName"),
            "date_verified": cr.get("datePublished"),
            "title": cr.get("name"),
            "source_url": item.get("url"),
            "speaker": author.get("name") if isinstance(author, dict) else None,
        })

    return rec

# -----------------------
# Main scraping logic
# -----------------------

def main():
    rows: List[Dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(locale="ro-RO")
        page = context.new_page()

        page.goto(LISTING_URL, wait_until="domcontentloaded")
        time.sleep(2)

        urls = set()

        for _ in range(MAX_LISTING_CLICKS):
            links = page.query_selector_all("a[href]")
            for a in links:
                href = a.get_attribute("href")
                if href and DOC_URL_RE.match(href):
                    urls.add(href)

            btn = page.query_selector("text=Vezi mai mult")
            if not btn:
                break
            btn.click()
            time.sleep(1.5)

        urls = sorted(urls)
        print(f"[INFO] Discovered {len(urls)} AFP articles")

        for url in tqdm(urls, desc="Scraping AFP articles"):
            try:
                page.goto(url, wait_until="domcontentloaded")
                time.sleep(REQUEST_DELAY)
                html = page.content()
                rec = parse_article(html, url)
                rows.append(rec)
            except Exception as e:
                rows.append({
                    "id": sha1(url),
                    "url": url,
                    "error": str(e),
                })

        browser.close()

    # -----------------------
    # Build DataFrame
    # -----------------------

    df = pd.DataFrame(rows)
    df["label_norm"] = df["label"].apply(normalize_label)
    df["label_group"] = df["label"].apply(label_group)
    df["claim_len"] = df["claim"].fillna("").apply(len)

    df_ok = df[
        df["claim"].notna() &
        df["label_group"].notna() &
        (df["claim_len"] >= 20)
    ].copy()

    # Deduplicate by claim text
    df_ok["claim_md5"] = df_ok["claim"].apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
    df_ok = df_ok.drop_duplicates("claim_md5").drop(columns="claim_md5")

    # -----------------------
    # Save outputs
    # -----------------------

    df.to_csv(OUT_DIR / "afp_verificat_raw.csv", index=False, encoding="utf-8")
    df_ok.to_csv(OUT_DIR / "afp_verificat_dataset.csv", index=False, encoding="utf-8")

    factual_like_cols = [
        "url", "type", "label", "label_group",
        "date_verified", "speaker", "claim", "source_url"
    ]
    df_ok[factual_like_cols].to_csv(
        OUT_DIR / "afp_verificat_dataset_factual_like.csv",
        index=False,
        encoding="utf-8"
    )

    print("[DONE] CSV datasets written to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
