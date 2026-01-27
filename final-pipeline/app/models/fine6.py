from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class Fine6Result:
    label: str
    probs: Dict[str, float]
    top_prob: float
    logits: list[float]


class Fine6Model:
    def __init__(self, model_dir: Path, labels: List[str], device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.labels = labels
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> Fine6Result:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Fine6Model not loaded")

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        logits = out.logits.detach().cpu().numpy()[0]
        probs_arr = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        probs = {}
        for i, lab in enumerate(self.labels):
            if i < len(probs_arr):
                probs[lab] = float(probs_arr[i])
            else:
                probs[lab] = 0.0

        best_idx = int(np.argmax(probs_arr))
        label = self.labels[best_idx] if best_idx < len(self.labels) else str(best_idx)
        top_prob = float(probs_arr[best_idx])

        return Fine6Result(label=label, probs=probs, top_prob=top_prob, logits=logits.tolist())
