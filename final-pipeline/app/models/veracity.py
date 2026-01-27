from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class VeracityResult:
    p_true: float
    logits: list[float]


class VeracityModel:
    def __init__(self, model_dir: Path, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, text: str) -> VeracityResult:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("VeracityModel not loaded")

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        logits = out.logits.detach().cpu().numpy()[0]
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        p_true = float(probs[1]) if probs.shape[0] > 1 else float(probs[0])
        return VeracityResult(p_true=p_true, logits=logits.tolist())
