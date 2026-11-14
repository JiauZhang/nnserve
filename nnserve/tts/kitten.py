import json
import os
import threading
from pathlib import Path

import numpy as np
import torch
from nnm.models.bart import BartForConditionalGeneration
from nnm.models.kitten_tts import KittenTTS, tokens_from_phonemes

from nnserve.handlers import SpeechAudio
from nnserve.tts.kitten_frontend import Frontend

GAP_S = 0.15
DEFAULT_LANGUAGE = "en"
OPENAI_VOICES = {
    "alloy": "eleanor",
    "ash": "otis",
    "ballad": "edith",
    "coral": "otis",
    "echo": "otis",
    "fable": "eleanor",
    "onyx": "otis",
    "nova": "eleanor",
    "sage": "otis",
    "shimmer": "eleanor",
    "verse": "edith",
    "marin": "eleanor",
    "cedar": "otis",
}


def resolve_voice(value):
    if isinstance(value, dict):
        value = value.get("id") or value.get("name") or ""
    name = str(value).strip()
    return OPENAI_VOICES.get(name.lower(), name)


class KittenTtsHandler:
    def __init__(self, entry):
        self.path = entry.path
        manifest = json.loads((Path(entry.path) / "nnm_weights.json").read_text())
        modules = manifest.get("modules", {})
        self.multilingual = "kitten_tts" in modules
        module = modules.get("kitten_tts") or (next(iter(modules.values())) if modules else None)
        self.config = KittenTTS.load_config(entry.path, module)
        self.sample_rate = int(self.config.sample_rate)
        self.vocab = json.loads((Path(entry.path) / "vocab.json").read_text(encoding="utf-8"))
        self.languages = sorted(
            f[len("voices_") : -len(".npz")] for f in os.listdir(entry.path) if f.startswith("voices_") and f.endswith(".npz")
        )
        self._packs = {}
        self._voice_index = {}
        for lang in sorted(self.languages, key=lambda code: (code != DEFAULT_LANGUAGE, code)):
            packs = np.load(Path(entry.path) / f"voices_{lang}.npz")
            self._packs[lang] = {k: packs[k] for k in packs.files}
            for key in packs.files:
                self._voice_index.setdefault(key.casefold(), (lang, key))
        bart = BartForConditionalGeneration.from_pretrained(str(Path(entry.path) / "g2p" / "g2p_bart_en_us")).eval()
        self.frontend = Frontend(self.config.phonemizer, self.vocab, str(Path(entry.path) / "g2p"), bart)
        self._models = {}
        self.lock = threading.Lock()

    def _model(self, lang):
        if lang not in self._models:
            variant = None if self.multilingual else lang
            self._models[lang] = KittenTTS.from_pretrained(self.path, variant=variant).eval()
        return self._models[lang]

    def _resolve(self, voice):
        name = resolve_voice(voice).casefold()
        if name not in self._voice_index:
            raise KeyError(f"voice {voice!r} is not available")
        return self._voice_index[name]

    def _chunk(self, lang, key, phonemes, speed, keep_start):
        tokens = tokens_from_phonemes(phonemes, self.vocab)
        pack = self._packs[lang][key]
        style = torch.from_numpy(pack[min(len(tokens) - 1, pack.shape[0] - 1)].copy()).float().unsqueeze(0)
        input_ids = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            result = self._model(lang).infer(input_ids, style, speed=speed)
        wav = np.clip(result["waveform"].numpy(), -1, 1)
        dur = result["duration"].numpy()
        total = int(dur.sum())
        spf = max(1, round(len(wav) / max(total, 1)))
        lead = 0 if keep_start else max(0, int(dur[0]) - 1) * spf
        trail = max(0, int(dur[-1]) - 1) * spf
        if lead + trail + spf < len(wav):
            wav = wav[lead : len(wav) - trail] if trail else wav[lead:]
        wav = np.ascontiguousarray(wav)
        fade = min(120, len(wav) // 8)
        if fade:
            wav[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            wav[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        return wav

    def synthesize(self, request, entry):
        lang, key = self._resolve(request.voice)
        speed = float(request.speed or 1.0)
        chunks, _ = self.frontend(request.input, lang)
        gap = np.zeros(int(GAP_S * self.sample_rate), dtype=np.float32)
        out = []
        with self.lock:
            for i, phonemes in enumerate(chunks):
                out.append(self._chunk(lang, key, phonemes, speed, keep_start=(i == 0)))
                out.append(gap)
        samples = np.concatenate(out[:-1]) if out else np.zeros(0, dtype=np.float32)
        return SpeechAudio(samples=samples, sample_rate=self.sample_rate)
