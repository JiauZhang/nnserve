import json


def encode_event(event):
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event['type']}\ndata: {payload}\n\n"


class EventSequence:
    def __init__(self):
        self._value = 0

    def next(self, event_type, **fields):
        self._value += 1
        return {"type": event_type, "sequence_number": self._value, **fields}
