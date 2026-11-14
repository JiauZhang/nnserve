import shutil

import pytest
from fastapi.testclient import TestClient

from nnserve.app import create_app
from nnserve.config import ModelConfig, ServeConfig


def test_kitten_tts_whistle_roundtrip(tts_model_path, asr_model_path):
    if tts_model_path is None or asr_model_path is None:
        pytest.skip("--tts-model-path and --asr-model-path required for the round-trip check")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required for audio decoding")

    config = ServeConfig(
        models=[
            ModelConfig(id="kitten-tts", path=str(tts_model_path), handler="nnserve.tts:KittenTtsHandler"),
            ModelConfig(id="whistle", path=str(asr_model_path), handler="nnserve.asr:WhistleAsrHandler"),
        ]
    )
    client = TestClient(create_app(config))

    speech = client.post(
        "/v1/audio/speech",
        json={"model": "kitten-tts", "input": "Hello world", "voice": "alloy", "response_format": "wav"},
    )
    assert speech.status_code == 200
    assert speech.headers["content-type"].startswith("audio/wav")
    assert speech.content.startswith(b"RIFF")

    transcription = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("speech.wav", speech.content, "audio/wav")},
        data={"model": "whistle"},
    )
    assert transcription.status_code == 200
    assert transcription.json()["text"].strip(" .!?").lower() == "hello world"


def test_qwen3_tts_qwen3_asr_roundtrip(qwen3_tts_model_path, qwen3_asr_model_path):
    if qwen3_tts_model_path is None or qwen3_asr_model_path is None:
        pytest.skip("--qwen3-tts-model-path and --qwen3-asr-model-path required for the round-trip check")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required for audio decoding")

    config = ServeConfig(
        models=[
            ModelConfig(id="qwen3-tts", path=str(qwen3_tts_model_path), handler="nnserve.tts:Qwen3TtsHandler"),
            ModelConfig(id="qwen3-asr", path=str(qwen3_asr_model_path), handler="nnserve.asr:Qwen3AsrHandler"),
        ]
    )
    client = TestClient(create_app(config))

    speech = client.post(
        "/v1/audio/speech",
        json={
            "model": "qwen3-tts",
            "input": "你好，世界。",
            "voice": "alloy",
            "language": "chinese",
            "response_format": "wav",
        },
    )
    assert speech.status_code == 200
    assert speech.content.startswith(b"RIFF")

    transcription = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("speech.wav", speech.content, "audio/wav")},
        data={"model": "qwen3-asr", "language": "chinese"},
    )
    assert transcription.status_code == 200
    assert "你好" in transcription.json()["text"]
