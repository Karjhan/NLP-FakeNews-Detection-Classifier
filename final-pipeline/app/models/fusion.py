from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd


def clamp(p: float, eps: float = 1e-6) -> float:
    return max(eps, min(1.0 - eps, float(p)))


def logit(p: float) -> float:
    p = clamp(p)
    return math.log(p / (1.0 - p))


@dataclass
class FusionResult:
    final_p_true: float
    threshold: float
    binary_label: str
    features: Dict[str, float]


class FusionModel:
    def __init__(
        self,
        model_path: Path,
        threshold_path: Path,
        feature_schema_path: Path,
    ):
        self.model_path = Path(model_path)
        self.threshold_path = Path(threshold_path)
        self.feature_schema_path = Path(feature_schema_path)

        self.model = None
        self.threshold = 0.5
        self.features: List[str] = []

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Fusion model not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

        if self.threshold_path.exists():
            obj = json.loads(self.threshold_path.read_text(encoding="utf-8"))
            self.threshold = float(obj.get("threshold", obj.get("best_threshold", 0.5)))

        if self.feature_schema_path.exists():
            obj = json.loads(self.feature_schema_path.read_text(encoding="utf-8"))
            self.features = list(obj.get("features", []))

        if not self.features:
            self.features = [
                "logit_p_true_content",
                "logit_p_not_clickbait",
                "source_score",
                "text_len",
                "has_source",
            ]

    def predict(
        self,
        *,
        p_true_content: float,
        p_clickbait: float,
        source_score: float,
        text_len: int,
        has_source: int,
    ) -> FusionResult:
        if self.model is None:
            raise RuntimeError("FusionModel not loaded")

        feat = {
            "logit_p_true_content": logit(float(p_true_content)),
            "logit_p_not_clickbait": logit(1.0 - float(p_clickbait)),
            "source_score": float(source_score),
            "text_len": float(text_len),
            "has_source": float(has_source),
        }

        X = pd.DataFrame([{k: feat[k] for k in self.features}])
        p = float(self.model.predict_proba(X)[:, 1][0])
        label = "TRUE" if p >= self.threshold else "FALSE"
        return FusionResult(final_p_true=p, threshold=self.threshold, binary_label=label, features=feat)
