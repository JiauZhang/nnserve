import io
import shutil
import subprocess
import wave

import numpy as np

MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

FFMPEG_ARGS = {
    "mp3": ["-f", "mp3", "-c:a", "libmp3lame", "-q:a", "2"],
    "opus": ["-f", "ogg", "-c:a", "libopus"],
    "aac": ["-f", "adts", "-c:a", "aac"],
    "flac": ["-f", "flac"],
}


def pcm_bytes(samples):
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def wav_bytes(samples, sample_rate):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm_bytes(samples))
    return buffer.getvalue()


def encode_audio(samples, sample_rate, response_format="mp3"):
    if response_format == "pcm":
        return pcm_bytes(samples), MEDIA_TYPES["pcm"]
    if response_format == "wav":
        return wav_bytes(samples, sample_rate), MEDIA_TYPES["wav"]
    if response_format not in FFMPEG_ARGS:
        raise ValueError(f"unsupported audio format {response_format!r}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"{response_format} output requires ffmpeg")

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *FFMPEG_ARGS[response_format], "pipe:1"],
        input=wav_bytes(samples, sample_rate),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "ignore").strip() or "ffmpeg failed")
    return result.stdout, MEDIA_TYPES[response_format]
