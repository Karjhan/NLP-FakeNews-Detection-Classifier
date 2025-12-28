from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from . import config
from .utils.text import build_text_input, text_len, normalize_ws
from .models.clickbait import ClickbaitModel
from .models.veracity import VeracityModel
from .models.fine6 import Fine6Model
from .models.fusion import FusionModel
from .models.source_prior import SourcePrior


@dataclass
class PipelineInput:
    title: Optional[str] = None
    claim: Optional[str] = None
    body: Optional[str] = None
    source_url: Optional[str] = None


class FakeNewsPipeline:
    def __init__(self):
        self.clickbait = ClickbaitModel(config.CLICKBAIT_MODEL_DIR, device=config.DEVICE)
        self.veracity = VeracityModel(config.VERACITY_MODEL_DIR, device=config.DEVICE)
        self.fine6 = Fine6Model(config.FINE6_MODEL_DIR, labels=config.FINE6_LABELS, device=config.DEVICE)

        self.fusion = FusionModel(
            model_path=config.FUSION_MODEL_PATH,
            threshold_path=config.FUSION_THRESHOLD_PATH,
            feature_schema_path=config.FUSION_FEATURE_SCHEMA_PATH,
        )

        self.source_prior = SourcePrior(
            table_csv=config.SOURCE_VERACITY_TABLE_PATH,
            platform_domains=config.PLATFORM_DOMAINS,
            platform_neutral=config.PLATFORM_NEUTRAL,
        )

        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self.clickbait.load()
        self.veracity.load()
        self.fine6.load()
        self.fusion.load()
        self.source_prior.load()
        self._loaded = True

    def predict(self, inp: PipelineInput) -> Dict[str, Any]:
        self.load()

        text_input = build_text_input(title=inp.title, claim=inp.claim, body=inp.body)
        tl = text_len(text_input)

        clickbait_text = normalize_ws(inp.title or "") or text_input
        cb = self.clickbait.predict_proba(clickbait_text)

        ver = self.veracity.predict_proba(text_input)

        sp = self.source_prior.lookup(inp.source_url or "")

        if (
                tl < config.NEUTRAL_MAX_TEXT_LEN
                and ver.p_true < config.NEUTRAL_CONTENT_MAX_P_TRUE
                and sp.p_true >= config.HIGH_TRUST_MIN_P_TRUE
        ):
            return {
                "input": {
                    "text_len": tl,
                    "source_url": inp.source_url or "",
                    "source_domain": sp.source_domain,
                },
                "component_outputs": {
                    "p_clickbait": cb.p_clickbait,
                    "p_true_content": ver.p_true,
                    "source_score": sp.source_score,
                    "p_true_source": sp.p_true,
                    "source_evidence": sp.evidence,
                },
                "fusion": {
                    "final_p_true": sp.p_true,
                    "threshold": self.fusion.threshold,
                    "binary_label": "TRUE",
                    "features": {
                        "neutral_override": True
                    },
                },
                "fine6": {
                    "fine6_label": "TRUE",
                    "raw_fine6_label": "TRUE",
                    "top_prob": 1.0,
                    "probs": {"TRUE": 1.0},
                },
                "gated": {
                    "gated_label": "TRUE",
                },
            }

        fusion_out = self.fusion.predict(
            p_true_content=ver.p_true,
            p_clickbait=cb.p_clickbait,
            source_score=sp.source_score,
            text_len=tl,
            has_source=1 if sp.source_domain else 0,
        )

        fine = self.fine6.predict(text_input)

        fine6_label = fine.label
        if fine.top_prob < config.INCONCLUSIVE_MIN_TOP_PROB:
            fine6_label = "INCONCLUSIVE"

        gated_label = self._gated_fine_label(
            fusion_binary=fusion_out.binary_label,
            fine_probs=fine.probs,
            top_prob=fine.top_prob,
            source_domain=sp.source_domain,
        )

        return {
            "input": {
                "text_len": tl,
                "source_url": inp.source_url or "",
                "source_domain": sp.source_domain,
            },
            "component_outputs": {
                "p_clickbait": cb.p_clickbait,
                "p_true_content": ver.p_true,
                "source_score": sp.source_score,
                "p_true_source": sp.p_true,
                "source_evidence": sp.evidence,
            },
            "fusion": {
                "final_p_true": fusion_out.final_p_true,
                "threshold": fusion_out.threshold,
                "binary_label": fusion_out.binary_label,
                "features": fusion_out.features,
            },
            "fine6": {
                "fine6_label": fine6_label,
                "raw_fine6_label": fine.label,
                "top_prob": fine.top_prob,
                "probs": fine.probs,
            },
            "gated": {
                "gated_label": gated_label,
            },
        }

    def _gated_fine_label(
            self,
            *,
            fusion_binary: str,
            fine_probs: Dict[str, float],
            top_prob: float,
            source_domain: str,
    ) -> str:
        if (
                fusion_binary == "FALSE"
                and source_domain in config.SATIRE_DOMAINS
        ):
            return "SATIRE"

        if (
                fusion_binary == "FALSE"
                and source_domain in config.PROPAGANDA_DOMAINS
                and fine_probs.get("PROPAGANDA", 0.0) >= 0.05
        ):
            return "PROPAGANDA"

        if top_prob < config.INCONCLUSIVE_MIN_TOP_PROB:
            return "INCONCLUSIVE"

        if fusion_binary == "FALSE" and fine_probs.get("PARTIAL TRUE", 0) > 0.6:
            return "INCONCLUSIVE"

        if fusion_binary == "TRUE":
            candidates = ["TRUE", "PARTIAL TRUE"]
        else:
            candidates = ["FALSE", "MISLEADING", "PROPAGANDA", "SATIRE"]

        best = None
        bestp = -1.0
        for c in candidates:
            p = float(fine_probs.get(c, 0.0))
            if p > bestp:
                bestp = p
                best = c
        return best or "INCONCLUSIVE"
