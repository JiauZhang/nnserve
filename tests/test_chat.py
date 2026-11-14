import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers

from nnserve.app import create_app
from nnserve.config import ModelConfig, ServeConfig
from tests.conftest import write_manifest
from tests.helpers import parse_events


class TinyChatModel(torch.nn.Module):
    def __init__(self, vocab_size=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def forward(self, input_ids, cache=None):
        if cache is not None:
            keys = input_ids.float().unsqueeze(-1)
            for layer in cache:
                layer.update(keys, keys)
        next_ids = (input_ids + 1) % self.vocab_size
        logits = torch.full((*input_ids.shape, self.vocab_size), -5.0, device=input_ids.device)
        logits.scatter_(-1, next_ids.unsqueeze(-1), 5.0)
        return logits


def make_client(tmp_path):
    path = write_manifest(
        tmp_path / "tiny-chat",
        "chat",
        [],
        model_class="tests.test_chat:TinyChatModel",
        chat_template="{{prompt}}",
    )
    tokenizer = Tokenizer(models.WordLevel({"a": 0, "b": 1, "c": 2}, unk_token="a"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))

    config = ServeConfig(models=[ModelConfig(id="tiny-chat", path=str(path), handler="nnserve.chat:ChatHandler")])
    return TestClient(create_app(config))


def test_chat_response_uses_the_engine(tmp_path):
    client = make_client(tmp_path)

    response = client.post("/v1/responses", json={"model": "tiny-chat", "input": "a", "max_output_tokens": 4})

    assert response.status_code == 200
    body = response.json()
    assert body["output_text"]
    assert body["usage"]["input_tokens"] == 1
    assert body["usage"]["output_tokens"] == 4


def test_chat_stream_matches_the_final_text(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/v1/responses",
        json={"model": "tiny-chat", "input": "a", "max_output_tokens": 4, "stream": True},
    )

    events = parse_events(response.text)
    deltas = [data["delta"] for event, data in events if event == "response.output_text.delta"]
    completed = events[-1][1]["response"]
    assert "".join(deltas) == completed["output_text"]
