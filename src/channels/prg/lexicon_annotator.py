"""
Multi-lexicon sentiment annotation pipeline.
Fuses polarity signals from NRC, Hu-Liu and SenticNet into
per-token polarity vectors with functional category assignment.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

_NEGATION_MARKERS = None
_INTENSIFIER_LIST = None

@dataclass
class TokenAnnotation:
    token: str
    index: int
    polarity_vec: np.ndarray
    category: str
    lexicon_hits: Dict[str, float] = field(default_factory=dict)

class MultiLexiconAnnotator:
    def __init__(self, nrc_path, huliu_path, senticnet_path, negation_path, intensifier_path, lowercase=True, oov_strategy="zero"):
        self._lowercase = lowercase
        self._oov_strategy = oov_strategy
        self._nrc = self._load_nrc(nrc_path)
        self._huliu_pos, self._huliu_neg = self._load_huliu(huliu_path)
        self._senticnet = self._load_senticnet(senticnet_path)
        global _NEGATION_MARKERS, _INTENSIFIER_LIST
        _NEGATION_MARKERS = self._load_wordlist(negation_path)
        _INTENSIFIER_LIST = self._load_wordlist(intensifier_path)

    @staticmethod
    def _load_wordlist(path):
        with open(path, "r", encoding="utf-8") as fh:
            return {line.strip().lower() for line in fh if line.strip()}

    def _load_nrc(self, path):
        mapping = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    word, cat, flag = parts[0], parts[1], int(parts[2])
                    if cat == "positive" and flag == 1:
                        mapping[word.lower()] = 1.0
                    elif cat == "negative" and flag == 1:
                        mapping.setdefault(word.lower(), -1.0)
        return mapping

    def _load_huliu(self, path):
        pos, neg = set(), set()
        for fname in ("positive-words.txt", "negative-words.txt"):
            target = pos if "positive" in fname else neg
            full = f"{path}/{fname}"
            try:
                with open(full, "r", encoding="latin-1") as fh:
                    for line in fh:
                        w = line.strip()
                        if w and not w.startswith(";"):
                            target.add(w.lower())
            except FileNotFoundError:
                pass
        return pos, neg

    def _load_senticnet(self, path):
        mapping = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        mapping[parts[0].lower().replace("_", " ")] = float(parts[1])
                    except ValueError:
                        continue
        return mapping

    def _get_polarity_vec(self, token):
        t = token.lower() if self._lowercase else token
        s_nrc = self._nrc.get(t, 0.0)
        s_huliu = 1.0 if t in self._huliu_pos else (-1.0 if t in self._huliu_neg else 0.0)
        s_sn = np.sign(self._senticnet.get(t, 0.0))
        return np.array([s_nrc, s_huliu, s_sn], dtype=np.float32)

    def _classify_token(self, token, polarity_vec):
        t = token.lower()
        if t in _NEGATION_MARKERS:
            return "negation"
        if t in _INTENSIFIER_LIST:
            return "intensifier"
        if np.any(polarity_vec != 0):
            return "sentiment"
        return "neutral"

    def annotate(self, tokens):
        annotations = []
        for idx, tok in enumerate(tokens):
            pvec = self._get_polarity_vec(tok)
            cat = self._classify_token(tok, pvec)
            annotations.append(TokenAnnotation(token=tok, index=idx, polarity_vec=pvec, category=cat))
        return annotations

    def coverage_stats(self, tokens):
        total = len([t for t in tokens if t.isalpha()])
        if total == 0:
            return {"nrc": 0.0, "huliu": 0.0, "senticnet": 0.0, "union": 0.0}
        hits = {"nrc": 0, "huliu": 0, "senticnet": 0, "union": 0}
        for t in tokens:
            if not t.isalpha():
                continue
            tl = t.lower()
            n = tl in self._nrc
            h = tl in self._huliu_pos or tl in self._huliu_neg
            s = tl in self._senticnet
            hits["nrc"] += int(n); hits["huliu"] += int(h)
            hits["senticnet"] += int(s); hits["union"] += int(n or h or s)
        return {k: v / total for k, v in hits.items()}
