import io
import json
import wave

import numpy as np

PNG_DATA_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def wav_bytes(seconds=0.5, sample_rate=16000):
    samples = np.sin(np.linspace(0, 2 * np.pi * 440 * seconds, int(seconds * sample_rate))).astype(np.float32)
    pcm = (np.clip(samples * 0.1, -1, 1) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def parse_events(body):
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line[len("event: ") :] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[len("data: ") :] for line in lines if line.startswith("data: ")))
        events.append((event, data))
    return events
