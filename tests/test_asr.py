import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nnserve.app import create_app
from nnserve.asr import decode_audio, render_transcription
from nnserve.config import ModelConfig, ServeConfig
from nnserve.handlers import AsrResult
from nnserve.schemas import TranscriptionRequest
from tests.helpers import wav_bytes


def test_decode_wav():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required for audio decoding")

    samples = decode_audio(wav_bytes())

    assert samples.ndim == 1
    assert samples.dtype == np.float32
    assert samples.size == 8000


def test_render_json_and_text():
    result = AsrResult(text="hello world", language="en")

    assert render_transcription(result, TranscriptionRequest(model="x"), 1.0) == (
        {"text": "hello world"},
        "application/json",
    )
    assert render_transcription(result, TranscriptionRequest(model="x", response_format="text"), 1.0) == (
        "hello world",
        "text/plain",
    )


def test_render_verbose_json_with_words():
    result = AsrResult(
        text="hello world",
        language="en",
        words=[{"word": "hello", "start": 0.0, "end": 0.4}, {"word": "world", "start": 0.5, "end": 0.9}],
    )
    request = TranscriptionRequest(model="x", response_format="verbose_json", timestamp_granularities=["word"])
    body, media_type = render_transcription(result, request, 1.0)

    assert media_type == "application/json"
    assert body["task"] == "transcribe"
    assert body["language"] == "en"
    assert body["duration"] == 1.0
    assert body["segments"][0]["end"] == 1.0
    assert [word["word"] for word in body["words"]] == ["hello", "world"]


def test_render_subtitles():
    result = AsrResult(text="hello world", language="en")

    srt, _ = render_transcription(result, TranscriptionRequest(model="x", response_format="srt"), 1.0)
    vtt, _ = render_transcription(result, TranscriptionRequest(model="x", response_format="vtt"), 1.0)

    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,000")
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.000")


def test_whistle_through_the_http_api(asr_model_path):
    if asr_model_path is None:
        pytest.skip("--asr-model-path required for the end-to-end ASR check")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required for audio decoding")

    config = ServeConfig(models=[ModelConfig(id="whistle", path=str(asr_model_path), handler="nnserve.asr:WhistleAsrHandler")])
    client = TestClient(create_app(config))
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", wav_bytes(1.0), "audio/wav")},
        data={"model": "whistle"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": ""}
