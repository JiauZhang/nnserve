import shutil
import subprocess

import numpy as np

TARGET_RATE = 16000


def decode_audio(data, sample_rate=TARGET_RATE):
    if not data:
        raise ValueError("empty audio upload")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("audio decoding requires ffmpeg")

    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-vn",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.decode("utf-8", "ignore").strip() or "could not decode audio")
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if samples.size == 0:
        raise ValueError("no audio samples decoded")
    return samples
