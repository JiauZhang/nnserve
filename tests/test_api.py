import shutil

import pytest

from tests.helpers import PNG_DATA_URI, parse_events, wav_bytes


class TestModels:
    def test_lists_registered_models(self, client):
        response = client.get("/v1/models")

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"
        assert [model["id"] for model in body["data"]] == [
            "fake-chat",
            "fake-text",
            "fake-ocr",
            "fake-tts",
            "fake-asr",
        ]
        assert {model["task"] for model in body["data"]} == {"chat", "ocr", "tts", "asr"}
        assert body["data"][0]["object"] == "model"

    def test_unknown_model_uses_openai_error(self, client):
        response = client.get("/v1/models/missing")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "model_not_found"


class TestResponses:
    def test_sync_response(self, client):
        response = client.post("/v1/responses", json={"model": "fake-chat", "input": "hi"})

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "response"
        assert body["status"] == "completed"
        assert body["output_text"] == "hello world"
        assert body["output"][0]["content"][0]["text"] == "hello world"
        assert body["usage"]["total_tokens"] == 5
        assert body["store"] is False
        assert body["text"]["format"]["type"] == "text"

    def test_instructions_only_is_allowed(self, client):
        response = client.post("/v1/responses", json={"model": "fake-chat", "instructions": "be brief"})

        assert response.status_code == 200

    def test_empty_request_is_rejected(self, client):
        response = client.post("/v1/responses", json={"model": "fake-chat"})

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "missing_required_parameter"

    def test_stream_uses_responses_events(self, client):
        response = client.post("/v1/responses", json={"model": "fake-chat", "input": "hi", "stream": True})

        assert response.status_code == 200
        events = parse_events(response.text)
        types = [event for event, _ in events]

        assert types[0] == "response.created"
        assert types[1] == "response.in_progress"
        assert "response.output_text.delta" in types
        assert types[-1] == "response.completed"
        assert "[DONE]" not in response.text
        assert [data["sequence_number"] for _, data in events] == list(range(1, len(events) + 1))

        deltas = [data["delta"] for event, data in events if event == "response.output_text.delta"]
        assert "".join(deltas) == "hello"
        completed = events[-1][1]["response"]
        assert completed["output_text"] == "hello"
        assert completed["usage"]["total_tokens"] == 5

    def test_tools_are_rejected(self, client):
        response = client.post(
            "/v1/responses",
            json={"model": "fake-chat", "input": "hi", "tools": [{"type": "function", "name": "f"}]},
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unsupported_parameter"
        assert response.json()["error"]["param"] == "tools"

    def test_structured_output_is_rejected(self, client):
        response = client.post(
            "/v1/responses",
            json={"model": "fake-chat", "input": "hi", "text": {"format": {"type": "json_schema"}}},
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unsupported_parameter"

    def test_image_requires_vision_capability(self, client):
        payload = {
            "model": "fake-text",
            "input": [{"role": "user", "content": [{"type": "input_image", "image_url": PNG_DATA_URI}]}],
        }
        response = client.post("/v1/responses", json=payload)

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unsupported_capability"

    def test_image_is_accepted_with_vision(self, client):
        payload = {
            "model": "fake-chat",
            "input": [{"role": "user", "content": [{"type": "input_image", "image_url": PNG_DATA_URI}]}],
        }
        response = client.post("/v1/responses", json=payload)

        assert response.status_code == 200


class TestOcr:
    def test_mistral_response_shape(self, client):
        response = client.post(
            "/v1/ocr",
            json={
                "model": "fake-ocr",
                "document": {"type": "image_url", "image_url": PNG_DATA_URI},
                "include_blocks": True,
                "confidence_scores_granularity": "page",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "fake-ocr"
        assert body["pages"][0]["index"] == 0
        assert body["pages"][0]["markdown"] == "page 0"
        assert body["pages"][0]["dimensions"] == {"dpi": 200.0, "height": 1, "width": 1}
        assert body["usage_info"]["pages_processed"] == 1
        assert body["usage_info"]["doc_size_bytes"] > 0

    def test_annotation_formats_are_rejected(self, client):
        response = client.post(
            "/v1/ocr",
            json={
                "model": "fake-ocr",
                "document": {"type": "image_url", "image_url": PNG_DATA_URI},
                "document_annotation_format": {"type": "json_schema"},
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unsupported_parameter"

    def test_page_selection_out_of_range(self, client):
        response = client.post(
            "/v1/ocr",
            json={
                "model": "fake-ocr",
                "document": {"type": "image_url", "image_url": PNG_DATA_URI},
                "pages": [1],
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_document"

    def test_task_mismatch(self, client):
        response = client.post(
            "/v1/ocr",
            json={"model": "fake-chat", "document": {"type": "image_url", "image_url": PNG_DATA_URI}},
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_supported"


class TestSpeech:
    def test_wav_output(self, client):
        response = client.post(
            "/v1/audio/speech",
            json={"model": "fake-tts", "input": "hello", "voice": "alloy", "response_format": "wav"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")
        assert response.content.startswith(b"RIFF")

    def test_pcm_output(self, client):
        response = client.post(
            "/v1/audio/speech",
            json={"model": "fake-tts", "input": "hello", "voice": "alloy", "response_format": "pcm"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/pcm"
        assert len(response.content) == 2400 * 2

    def test_mp3_output(self, client):
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg required for mp3 output")

        response = client.post(
            "/v1/audio/speech",
            json={"model": "fake-tts", "input": "hello", "voice": "alloy"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert len(response.content) > 0

    def test_unknown_voice(self, client):
        response = client.post("/v1/audio/speech", json={"model": "fake-tts", "input": "hello", "voice": "unknown"})

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "voice_not_found"

    def test_task_mismatch(self, client):
        response = client.post("/v1/audio/speech", json={"model": "fake-chat", "input": "hello", "voice": "alloy"})

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_supported"

    def test_sse_stream_is_rejected(self, client):
        response = client.post(
            "/v1/audio/speech",
            json={"model": "fake-tts", "input": "hello", "voice": "alloy", "stream_format": "sse"},
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unsupported_parameter"

    def test_invalid_speed(self, client):
        response = client.post(
            "/v1/audio/speech",
            json={"model": "fake-tts", "input": "hello", "voice": "alloy", "speed": 10},
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required for audio decoding")
class TestTranscriptions:
    def test_json_output(self, client):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", wav_bytes(), "audio/wav")},
            data={"model": "fake-asr"},
        )

        assert response.status_code == 200
        assert response.json() == {"text": "hello world"}

    def test_text_output(self, client):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", wav_bytes(), "audio/wav")},
            data={"model": "fake-asr", "response_format": "text"},
        )

        assert response.status_code == 200
        assert response.text == "hello world"

    def test_verbose_json_with_word_timestamps(self, client):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", wav_bytes(), "audio/wav")},
            data={
                "model": "fake-asr",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["text"] == "hello world"
        assert body["language"] == "en"
        assert body["duration"] == pytest.approx(0.5, abs=0.1)
        assert body["segments"]
        assert [word["word"] for word in body["words"]] == ["hello", "world"]

    def test_srt_output(self, client):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", wav_bytes(), "audio/wav")},
            data={"model": "fake-asr", "response_format": "srt"},
        )

        assert response.status_code == 200
        assert "00:00:00,000 --> 00:00:00,500" in response.text

    def test_vtt_output(self, client):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", wav_bytes(), "audio/wav")},
            data={"model": "fake-asr", "response_format": "vtt"},
        )

        assert response.status_code == 200
        assert response.text.startswith("WEBVTT")

    def test_task_mismatch(self, client):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", wav_bytes(), "audio/wav")},
            data={"model": "fake-chat"},
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_supported"

    def test_invalid_response_format(self, client):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", wav_bytes(), "audio/wav")},
            data={"model": "fake-asr", "response_format": "ogg"},
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"

    def test_invalid_audio(self, client):
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.bin", b"not audio", "application/octet-stream")},
            data={"model": "fake-asr"},
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_audio"


class TestService:
    def test_health_and_metrics(self, client):
        assert client.get("/health").json() == {"status": "ok"}
        metrics = client.get("/metrics").text
        assert "nnserve_models 5" in metrics

    def test_bearer_token(self, secured_client):
        assert secured_client.get("/health").status_code == 200
        assert secured_client.get("/v1/models").status_code == 401
        assert secured_client.get("/v1/models").json()["error"]["code"] == "invalid_api_key"

        authorized = secured_client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        assert authorized.status_code == 200
