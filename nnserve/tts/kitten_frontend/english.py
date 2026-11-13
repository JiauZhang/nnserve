"""English G2P for the model's phoneme vocab: tts_norm -> misaki lexicon -> CMUdict -> nnm BART G2P.

Vendored from kittenml's English frontend; the BART fallback is loaded through nnm instead of
transformers, and the Linux-only no-espeak assertions are dropped (nnserve uses espeak elsewhere).
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field

import torch

DIGITS = "zero one two three four five six seven eight nine".split()
TRANSLIT = str.maketrans(
    {
        "ø": "o",
        "Ø": "O",
        "ß": "ss",
        "æ": "ae",
        "Æ": "AE",
        "œ": "oe",
        "Œ": "OE",
        "ł": "l",
        "Ł": "L",
        "đ": "d",
        "Đ": "D",
        "’": "'",
    }
)


class Fallback:
    """misaki `fallback=` hook: CMUdict first, the nnm BART G2P for the rest."""

    RATING = 2

    def __init__(self, bart_model, cmu_lexicon: str):
        with open(cmu_lexicon, encoding="utf-8") as f:
            self.cmu = json.load(f)
        self.model = bart_model
        cfg = bart_model.config
        self.g2i = {c: i for i, c in enumerate(cfg.grapheme_chars) if i > 3}
        self.i2p = {i: c for i, c in enumerate(cfg.phoneme_chars) if i > 3}
        self.max_len = cfg.max_position_embeddings
        letters = re.escape("".join(self.g2i))
        self._pieces = re.compile(rf"(\d+)|([{letters}]+)|(.)", re.S)
        self._memo: dict[str, str] = {}
        self.used: list[tuple[str, str, str]] = []
        self.skipped: list[str] = []

    def reset(self) -> None:
        self.used, self.skipped = [], []

    def _bart(self, word: str) -> str:
        if word not in self._memo:
            out = []
            step = self.max_len - 2
            for i in range(0, len(word), step):
                ids = torch.tensor([[1] + [self.g2i[c] for c in word[i : i + step]] + [2]])
                with torch.no_grad():
                    gen = self.model.generate(ids, max_length=self.max_len)
                out.append("".join(self.i2p.get(int(t), "") for t in gen[0]))
            self._memo[word] = "".join(out).replace("ɾ", "T").replace("ʔ", "t")
        return self._memo[word]

    def _cmu(self, word: str):
        w = word.lower()
        if w in self.cmu:
            return self.cmu[w]
        if w.endswith("'s") and w[:-2] in self.cmu:
            stem = self.cmu[w[:-2]]
            return stem + ("s" if stem[-1] in "ptkfθ" else "ᵻz" if stem[-1] in "szʃʒʧʤ" else "z")
        return None

    def _word(self, word: str, sources: set) -> str:
        ps = self._cmu(word)
        if ps is None and "-" in word:
            return " ".join(self._word(p, sources) for p in word.split("-") if p)
        sources.add("CMUdict" if ps else "BART")
        return ps or self._bart(word)

    def say(self, text: str):
        t = unicodedata.normalize("NFKD", text.translate(TRANSLIT))
        t = "".join(c for c in t if not unicodedata.combining(c))
        out, sources = [], set()
        for digits, letters, other in self._pieces.findall(t):
            if digits:
                out.append(" ".join(self._word(DIGITS[int(d)], sources) for d in digits))
            elif letters.strip("'-."):
                out.append(self._word(letters.strip("'-."), sources))
            elif other.strip():
                self.skipped.append(other)
        return " ".join(p for p in out if p), "+".join(sorted(sources))

    def __call__(self, token):
        ps, source = self.say(token.text)
        if not ps:
            self.skipped.append(token.text)
            return None, None
        self.used.append((token.text, ps, source))
        return ps, self.RATING


@dataclass
class G2PResult:
    text: str
    phonemes: str
    fallback: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    dropped: list = field(default_factory=list)


class EnglishG2P:
    """tts_norm -> misaki lexicon -> CMUdict -> nnm BART -> the model's phoneme vocab."""

    def __init__(self, vocab: dict, g2p_dir: str, bart_model):
        from . import phonemize_v1 as pm
        from misaki import en
        from .tts_norm import normalize

        self._pm, self._normalize = pm, normalize
        self.vocab = dict(vocab)
        self.fallback = Fallback(bart_model, os.path.join(g2p_dir, "cmu_misaki_us.json"))
        self.g2p = en.G2P(trf=False, british=False, fallback=self.fallback, unk="")
        self._unsaid: list[str] = []

    def _run(self, text: str):
        ps, tokens = self.g2p(text)
        self._unsaid = [tk.text for tk in tokens if tk.phonemes is None]
        return ps, tokens

    def __call__(self, text: str, normalise: bool = True) -> G2PResult:
        text = self._normalize(text) if normalise else text
        self.fallback.reset()
        ps = self._pm.phonemize_lang(self._run, text, "en")
        dropped = sorted({c for c in ps if c not in self.vocab})
        if dropped:
            ps = "".join(c for c in ps if c in self.vocab)
        skipped = list(dict.fromkeys(self.fallback.skipped + self._unsaid))
        return G2PResult(text, ps, list(self.fallback.used), skipped, dropped)
