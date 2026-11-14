import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nnserve.app import create_app
from nnserve.config import ModelConfig, ServeConfig
from nnserve.tts import encode_audio, resolve_speaker, resolve_voice


def test_openai_voices_map_to_kitten_voices():
    assert resolve_voice("alloy") == "eleanor"
    assert resolve_voice({"id": "nova"}) == "eleanor"
    assert resolve_voice("Eleanor") == "Eleanor"
    assert resolve_voice("otis") == "otis"


def test_qwen3_openai_voices_map_to_native_speakers():
    speakers = frozenset({"serena", "ryan", "vivian"})

    assert resolve_speaker("alloy", speakers) == "serena"
    assert resolve_speaker({"id": "ash"}, speakers) == "ryan"
    assert resolve_speaker("Vivian", speakers) == "vivian"
    with pytest.raises(KeyError):
        resolve_speaker("bruno", speakers)


def test_encode_wav_and_pcm():
    samples = np.zeros(2400, np.float32)
    wav, media_type = encode_audio(samples, 24000, "wav")
    pcm, pcm_type = encode_audio(samples, 24000, "pcm")

    assert wav.startswith(b"RIFF")
    assert media_type == "audio/wav"
    assert len(pcm) == 2400 * 2
    assert pcm_type == "audio/pcm"


def test_encode_mp3_requires_ffmpeg():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required for mp3 output")

    audio, media_type = encode_audio(np.zeros(2400, np.float32), 24000, "mp3")

    assert len(audio) > 0
    assert media_type == "audio/mpeg"


def test_kitten_tts_through_the_http_api(tts_model_path):
    if tts_model_path is None:
        pytest.skip("--tts-model-path required for the end-to-end TTS check")

    config = ServeConfig(
        models=[ModelConfig(id="kitten-tts", path=str(tts_model_path), handler="nnserve.tts:KittenTtsHandler")]
    )
    client = TestClient(create_app(config))
    response = client.post(
        "/v1/audio/speech",
        json={"model": "kitten-tts", "input": "Hello world", "voice": "alloy", "response_format": "wav"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content.startswith(b"RIFF")
    assert len(response.content) > 1000


@pytest.mark.parametrize("language,text", [("english", "Hello."), ("chinese", "你好，世界。")])
def test_qwen3_tts_through_the_http_api(qwen3_tts_model_path, language, text):
    if qwen3_tts_model_path is None:
        pytest.skip("--qwen3-tts-model-path required for the end-to-end Qwen3-TTS check")

    config = ServeConfig(
        models=[ModelConfig(id="qwen3-tts", path=str(qwen3_tts_model_path), handler="nnserve.tts:Qwen3TtsHandler")]
    )
    client = TestClient(create_app(config))
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "qwen3-tts",
            "input": text,
            "voice": "alloy",
            "language": language,
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content.startswith(b"RIFF")
    assert len(response.content) > 1000
