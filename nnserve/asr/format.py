def timestamp(seconds, decimal=False):
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    seconds_part, milliseconds = divmod(remainder, 1000)
    separator = "." if decimal else ","
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}{separator}{milliseconds:03d}"


def render_transcription(result, request, duration):
    response_format = request.response_format
    if response_format == "text":
        return result.text, "text/plain"
    if response_format == "json":
        return {"text": result.text}, "application/json"

    if response_format == "verbose_json":
        segments = (
            [
                {
                    "id": 0,
                    "seek": 0,
                    "start": 0.0,
                    "end": round(duration, 2),
                    "text": result.text,
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0,
                }
            ]
            if result.text
            else []
        )
        body = {
            "task": "transcribe",
            "language": result.language,
            "duration": round(duration, 2),
            "text": result.text,
            "segments": segments,
        }
        if "word" in request.timestamp_granularities:
            body["words"] = [{"word": word["word"], "start": word["start"], "end": word["end"]} for word in result.words]
        return body, "application/json"

    cue = f"1\n{timestamp(0)} --> {timestamp(duration)}\n{result.text}\n"
    if response_format == "srt":
        return cue, "text/plain"
    return f"WEBVTT\n\n{timestamp(0, decimal=True)} --> {timestamp(duration, decimal=True)}\n{result.text}\n", "text/plain"
