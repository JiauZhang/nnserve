import threading

import torch
from nnm.models.whistle.model import LANGUAGES, Whistle

from nnserve.handlers import AsrResult


class WhistleAsrHandler:
    def __init__(self, entry):
        self.model = Whistle.from_pretrained(entry.path)
        self.lock = threading.Lock()

    def transcribe(self, request, entry, audio):
        if request.language and request.language not in LANGUAGES:
            raise ValueError(f"language {request.language!r} is not supported")
        word_timestamps = "word" in request.timestamp_granularities
        with self.lock:
            result = self.model.transcribe(
                torch.from_numpy(audio),
                language=request.language,
                word_timestamps=word_timestamps,
                beam=5,
            )
        return AsrResult(
            text=result["text"],
            language=result["language"],
            words=result.get("words", []),
        )
