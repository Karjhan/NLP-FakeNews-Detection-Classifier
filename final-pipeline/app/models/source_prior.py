from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.domains import get_domain, normalize_domain, is_platform_domain

@dataclass
class SourcePriorResult:
    source_domain: str
    source_score: float
    p_true: float
    evidence: str


class SourcePrior:
    def __init__(
        self,
        table_csv: Path,
        *,
        platform_domains: set[str],
        platform_neutral: bool = True,
        default_score: float = 0.0,
        default_p_true: float = 0.5,
    ):
        self.table_csv = Path(table_csv)
        self.platform_domains = platform_domains
        self.platform_neutral = platform_neutral
        self.default_score = float(default_score)
        self.default_p_true = float(default_p_true)

        self._table = None  # type: Optional[pd.DataFrame]
        self._map_score = {}
        self._map_p_true = {}
        self._map_evidence = {}

    def load(self) -> None:
        if not self.table_csv.exists():
            self._table = pd.DataFrame()
            return

        df = pd.read_csv(self.table_csv, encoding="utf-8")
        df["source_domain"] = df["source_domain"].fillna("").astype(str).str.lower().str.replace("www.", "", regex=False)

        if "source_score_final" not in df.columns:
            raise ValueError("source veracity table missing 'source_score_final'")
        if "p_true_final" not in df.columns:
            raise ValueError("source veracity table missing 'p_true_final'")

        self._table = df
        self._map_score = dict(zip(df["source_domain"], df["source_score_final"]))
        self._map_p_true = dict(zip(df["source_domain"], df["p_true_final"]))
        self._map_evidence = dict(zip(df["source_domain"], df.get("evidence", ["unknown"] * len(df))))

    def lookup(self, source_url: str) -> SourcePriorResult:
        domain = normalize_domain(get_domain(source_url))
        if not domain:
            return SourcePriorResult("", self.default_score, self.default_p_true, "no-source")

        if self.platform_neutral and is_platform_domain(domain, self.platform_domains):
            return SourcePriorResult(domain, 0.0, 0.5, "platform-neutral")

        score = float(self._map_score.get(domain, self.default_score))
        p_true = float(self._map_p_true.get(domain, self.default_p_true))
        evidence = str(self._map_evidence.get(domain, "default"))
        return SourcePriorResult(domain, score, p_true, evidence)
