from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from nnserve.documents import PageImage
    from nnserve.registry import ModelEntry
    from nnserve.schemas import (
        OcrPage,
        OcrRequest,
        ResponsesRequest,
        SpeechRequest,
        TranscriptionRequest,
    )


@dataclass(frozen=True)
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0


@dataclass(frozen=True)
class ResponsesOutput:
    text: str
    usage: UsageStats = field(default_factory=UsageStats)


@dataclass(frozen=True)
class ResponsesChunk:
    text: str = ""
    usage: UsageStats | None = None


@dataclass(frozen=True)
class OcrResult:
    pages: list[OcrPage]
    document_annotation: str | None = None


@dataclass(frozen=True)
class SpeechAudio:
    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class AsrResult:
    text: str
    language: str = ""
    words: list[dict] = field(default_factory=list)


class ResponsesHandler(Protocol):
    def respond(self, request: ResponsesRequest, entry: ModelEntry) -> ResponsesOutput: ...

    def stream(self, request: ResponsesRequest, entry: ModelEntry) -> Iterator[ResponsesChunk]: ...


class OcrHandler(Protocol):
    def process(self, request: OcrRequest, entry: ModelEntry, pages: list[PageImage]) -> OcrResult: ...


class TtsHandler(Protocol):
    def synthesize(self, request: SpeechRequest, entry: ModelEntry) -> SpeechAudio: ...


class AsrHandler(Protocol):
    def transcribe(self, request: TranscriptionRequest, entry: ModelEntry, audio: np.ndarray) -> AsrResult: ...
