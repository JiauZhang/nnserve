import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nnserve.app import create_app
from nnserve.config import ModelConfig, ServeConfig
from nnserve.handlers import (
    AsrResult,
    OcrResult,
    ResponsesChunk,
    ResponsesOutput,
    SpeechAudio,
    UsageStats,
)
from nnserve.schemas import OcrDimensions, OcrPage


def pytest_addoption(parser):
    parser.addoption("--model-path", action="store", default=None, help="path to a prepared nnm weight repository")
    parser.addoption("--tts-model-path", action="store", default=None, help="path to a prepared TTS weight repository")
    parser.addoption(
        "--qwen3-tts-model-path", action="store", default=None, help="path to a prepared Qwen3-TTS weight repository"
    )
    parser.addoption("--asr-model-path", action="store", default=None, help="path to a prepared ASR weight repository")
    parser.addoption(
        "--qwen3-asr-model-path", action="store", default=None, help="path to a prepared Qwen3-ASR weight repository"
    )


@pytest.fixture(scope="session")
def model_path(request):
    return request.config.getoption("--model-path")


@pytest.fixture(scope="session")
def tts_model_path(request):
    return request.config.getoption("--tts-model-path")


@pytest.fixture(scope="session")
def qwen3_tts_model_path(request):
    return request.config.getoption("--qwen3-tts-model-path")


@pytest.fixture(scope="session")
def asr_model_path(request):
    return request.config.getoption("--asr-model-path")


@pytest.fixture(scope="session")
def qwen3_asr_model_path(request):
    return request.config.getoption("--qwen3-asr-model-path")


class FakeResponsesHandler:
    def respond(self, request, entry):
        return ResponsesOutput("hello world", UsageStats(input_tokens=3, output_tokens=2))

    def stream(self, request, entry):
        yield ResponsesChunk("hel")
        yield ResponsesChunk("lo", UsageStats(input_tokens=3, output_tokens=2))


class FakeOcrHandler:
    def process(self, request, entry, pages):
        return OcrResult(
            pages=[
                OcrPage(
                    index=page.index,
                    markdown=f"page {page.index}",
                    dimensions=OcrDimensions(dpi=page.dpi, height=page.height, width=page.width),
                )
                for page in pages
            ]
        )


class FakeTtsHandler:
    def synthesize(self, request, entry):
        if request.voice not in ("alloy", "Bruno"):
            raise KeyError(f"unknown voice {request.voice!r}")
        samples = np.sin(np.linspace(0, 2 * np.pi * 440 * 0.1, 2400)).astype(np.float32)
        return SpeechAudio(samples=samples, sample_rate=24000)


class FakeAsrHandler:
    def transcribe(self, request, entry, audio):
        return AsrResult(
            text="hello world",
            language="en",
            words=[
                {"word": "hello", "start": 0.0, "end": 0.4, "probability": 0.9},
                {"word": "world", "start": 0.5, "end": 0.9, "probability": 0.8},
            ],
        )


def write_manifest(path, task, capabilities, **extra):
    path.mkdir()
    (path / "nnm_weights.json").write_text(
        json.dumps(
            {
                "version": 1,
                "sources": [],
                "weights": ["model.safetensors"],
                "task": task,
                "capabilities": capabilities,
                **extra,
            }
        )
    )
    return path


@pytest.fixture
def models(tmp_path):
    return {
        "chat": write_manifest(tmp_path / "chat", "chat", ["vision"]),
        "text": write_manifest(tmp_path / "text", "chat", []),
        "ocr": write_manifest(tmp_path / "ocr", "ocr", []),
        "tts": write_manifest(tmp_path / "tts", "tts", []),
        "asr": write_manifest(tmp_path / "asr", "asr", []),
    }


@pytest.fixture
def client(models):
    config = ServeConfig(
        models=[
            ModelConfig(id="fake-chat", path=str(models["chat"])),
            ModelConfig(id="fake-text", path=str(models["text"])),
            ModelConfig(id="fake-ocr", path=str(models["ocr"])),
            ModelConfig(id="fake-tts", path=str(models["tts"])),
            ModelConfig(id="fake-asr", path=str(models["asr"])),
        ]
    )
    handlers = {
        "fake-chat": FakeResponsesHandler(),
        "fake-text": FakeResponsesHandler(),
        "fake-ocr": FakeOcrHandler(),
        "fake-tts": FakeTtsHandler(),
        "fake-asr": FakeAsrHandler(),
    }
    return TestClient(create_app(config, handlers=handlers))


@pytest.fixture
def secured_client(models):
    config = ServeConfig(
        models=[ModelConfig(id="fake-chat", path=str(models["chat"]))],
        api_key="secret",
    )
    return TestClient(create_app(config, handlers={"fake-chat": FakeResponsesHandler()}))
