import threading
from pathlib import Path

import torch
from tokenizers import Tokenizer

from nnm.models.qwen3_asr import Qwen3ASR
from nnm.models.qwen3_asr.audio import feat_extract_lengths
from nnserve.handlers import AsrResult

PROMPT_PREFIX = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>"
PROMPT_SUFFIX = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"


class Qwen3AsrHandler:
    def __init__(self, entry):
        try:
            from transformers import WhisperFeatureExtractor  # noqa: PLC0415
        except ImportError as error:
            raise RuntimeError("Qwen3-ASR requires the `asr` extra: pip install 'nnserve[asr]'") from error

        self.model = Qwen3ASR.from_pretrained(entry.path).eval()
        self.tokenizer = Tokenizer.from_file(str(Path(entry.path) / "tokenizer.json"))
        self.extractor = WhisperFeatureExtractor.from_pretrained(entry.path)
        self.lock = threading.Lock()

    def transcribe(self, request, entry, audio):
        processed = self.extractor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        features = processed["input_features"]
        mask = processed["attention_mask"]
        audio_length = int(feat_extract_lengths(mask.sum(-1))[0])
        prompt = PROMPT_PREFIX + "<|audio_pad|>" * audio_length + PROMPT_SUFFIX
        input_ids = torch.tensor([self.tokenizer.encode(prompt).ids], dtype=torch.long)

        with self.lock, torch.no_grad():
            tokens = self.model.generate(input_ids, features, mask, max_new_tokens=256)

        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return AsrResult(text=text, language=request.language or "")
