from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Set

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = Path(
    os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts"))
).resolve()

CLICKBAIT_MODEL_DIR = ARTIFACTS_DIR / "clickbait" / "rocloco_roberta_clickbait"
VERACITY_MODEL_DIR = ARTIFACTS_DIR / "binary" / "veracity_roberta"
FINE6_MODEL_DIR = ARTIFACTS_DIR / "multiclass" / "fine6_roberta"

FUSION_DIR = ARTIFACTS_DIR / "fusion"
FUSION_MODEL_PATH = FUSION_DIR / "fusion_lr.joblib"
FUSION_THRESHOLD_PATH = FUSION_DIR / "fusion_threshold.json"
FUSION_FEATURE_SCHEMA_PATH = FUSION_DIR / "fusion_feature_schema.json"

SOURCE_VERACITY_DIR = ARTIFACTS_DIR / "source_veracity"
SOURCE_VERACITY_TABLE_PATH = SOURCE_VERACITY_DIR / "source_veracity_table.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

FINE6_LABELS: List[str] = [
    "TRUE",
    "FALSE",
    "PARTIAL TRUE",
    "MISLEADING",
    "PROPAGANDA",
    "SATIRE",
]

FINE6_TO_BINARY: Dict[str, int] = {
    "TRUE": 1,
    "PARTIAL TRUE": 1,
    "FALSE": 0,
    "MISLEADING": 0,
    "PROPAGANDA": 0,
    "SATIRE": 0,
}

INCONCLUSIVE_MIN_TOP_PROB = float(
    os.getenv("INCONCLUSIVE_MIN_TOP_PROB", "0.45")
)

NEUTRAL_MAX_TEXT_LEN = int(os.getenv("NEUTRAL_MAX_TEXT_LEN", "120"))
NEUTRAL_CONTENT_MAX_P_TRUE = float(os.getenv("NEUTRAL_CONTENT_MAX_P_TRUE", "0.05"))
HIGH_TRUST_MIN_P_TRUE = float(os.getenv("HIGH_TRUST_MIN_P_TRUE", "0.80"))


PLATFORM_DOMAINS: Set[str] = {
    "facebook.com", "m.facebook.com",
    "instagram.com",
    "whatsapp.com", "wa.me",

    "youtube.com", "youtu.be",
    "tiktok.com",
    "rumble.com",
    "bitchute.com",
    "odysee.com",

    "twitter.com", "x.com",
    "reddit.com",
    "telegram.org", "t.me",
    "discord.com", "discord.gg",
    "vk.com",
    "ok.ru",

    "medium.com",
    "substack.com",
    "wordpress.com",
    "blogspot.com",
    "tumblr.com",
    "quora.com",
}

PLATFORM_NEUTRAL = True

HIGH_TRUST_DOMAINS: Set[str] = {
    "gov.ro",
    "agerpres.ro",
    "mapn.ro",
    "mai.gov.ro",
    "ms.ro",

    "europa.eu",
    "ec.europa.eu",
    "ema.europa.eu",
    "ecdc.europa.eu",
    "consilium.europa.eu",

    "who.int",
    "cdc.gov",
    "nih.gov",
    "fda.gov",
    "nhs.uk",

    "un.org",
    "nato.int",
    "worldbank.org",
    "imf.org",
    "oecd.org",

    "gov.uk",
    "usa.gov",
    "canada.ca",
    "gouv.fr",
    "bund.de",

    "veridica.ro",
    "factcheck.org",
    "politifact.com",
    "snopes.com",
    "reuters.com",
    "apnews.com",
}

SATIRE_DOMAINS: Set[str] = {
    "timesnewroman.ro",
    "catavencii.ro",
    "academiacatavencu.com",
    "kamikazeonline.ro",

    "theonion.com",
    "babylonbee.com",
    "thedailymash.co.uk",
    "waterfordwhispersnews.com",
    "thebeaverton.com",
    "clickhole.com",
}

PROPAGANDA_DOMAINS: Set[str] = {
    "activenews.ro",
    "national.ro",
    "flux24.ro",
    "solidnews.ro",
    "ortodoxinfo.ro",
    "stiripesurse.ro",
    "r3media.ro",

    "sputniknews.com",
    "ria.ru",
    "tass.ru",
    "rt.com",
    "pravda.ru",
    "news-pravda.com",
    "topwar.ru",

    "infowars.com",
    "globalresearch.ca",
    "naturalnews.com",
    "newspunch.com",
    "beforeitsnews.com",
    "oann.com",
    "theepochtimes.com",
    "breitbart.com",
}
