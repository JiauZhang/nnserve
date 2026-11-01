import threading
from pathlib import Path

import numpy as np
import torch
from nnm.models.qwen3_tts import Qwen3TTS, Qwen3TTSCodec
from tokenizers import Tokenizer

from nnserve.handlers import SpeechAudio

OPENAI_VOICES = {
    "alloy": "serena",
    "ash": "ryan",
    "ballad": "vivian",
    "coral": "dylan",
    "echo": "eric",
    "fable": "sohee",
    "onyx": "uncle_fu",
    "nova": "ono_anna",
    "sage": "aiden",
    "shimmer": "vivian",
    "verse": "serena",
    "marin": "ono_anna",
    "cedar": "ryan",
}


def resolve_speaker(value, speakers):
    if isinstance(value, dict):
        value = value.get("id") or value.get("name") or ""
    name = str(value).strip().lower()
    speaker = OPENAI_VOICES.get(name, name)
    if speaker not in speakers:
        raise KeyError(f"voice {value!r} is not available")
    return speaker


class Qwen3TtsHandler:
    def __init__(self, entry):
        self.talker = Qwen3TTS.from_pretrained(entry.path).eval()
        self.codec = Qwen3TTSCodec.from_pretrained(entry.path).eval()
        self.tokenizer = Tokenizer.from_file(str(Path(entry.path) / "tokenizer.json"))
        self.speakers = frozenset(self.talker.config.talker_config.spk_id)
        self.languages = frozenset(self.talker.config.talker_config.codec_language_id)
        self.sample_rate = int(self.codec.config.output_sample_rate)
        self.lock = threading.Lock()

    def synthesize(self, request, entry):
        speaker = resolve_speaker(request.voice, self.speakers)
        language = (request.language or "auto").lower()
        if language != "auto" and language not in self.languages:
            raise ValueError(f"language {language!r} is not supported")

        text = f"<|im_start|>assistant\n{request.input}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = torch.tensor([self.tokenizer.encode(text).ids], dtype=torch.long)
        with self.lock, torch.no_grad():
            codes = self.talker.generate_codes(
                input_ids,
                speaker,
                language,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                subtalker_do_sample=False,
            )
            if codes.shape[0] == 0:
                raise ValueError("the model produced no audio")
            waveform = self.codec.decode(codes)

        samples = np.asarray(waveform[0, 0].float().cpu(), np.float32)
        return SpeechAudio(samples=samples, sample_rate=self.sample_rate)
