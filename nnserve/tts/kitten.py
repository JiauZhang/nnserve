import threading

import numpy as np
from nnm.models.kitten_tts import KittenTTS

from nnserve.handlers import SpeechAudio

OPENAI_VOICES = {
    "alloy": "Bella",
    "ash": "Jasper",
    "ballad": "Luna",
    "coral": "Leo",
    "echo": "Jasper",
    "fable": "Luna",
    "onyx": "Bruno",
    "nova": "Rosie",
    "sage": "Hugo",
    "shimmer": "Kiki",
    "verse": "Rosie",
    "marin": "Bella",
    "cedar": "Bruno",
}


def resolve_voice(value):
    if isinstance(value, dict):
        value = value.get("id") or value.get("name") or ""
    name = str(value).strip()
    return OPENAI_VOICES.get(name, name or "Bruno")


class KittenTtsHandler:
    def __init__(self, entry):
        self.model = KittenTTS.from_pretrained(entry.path).eval()
        self.lock = threading.Lock()

    def synthesize(self, request, entry):
        voice = resolve_voice(request.voice)
        with self.lock:
            result = self.model.synthesize(request.input, voice=voice, speed=request.speed)
        return SpeechAudio(
            samples=np.asarray(result["waveform"], np.float32),
            sample_rate=int(result["sample_rate"]),
        )
